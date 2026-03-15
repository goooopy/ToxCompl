# Import NumPy and PyTorch
import numpy as np
import torch
import pandas as pd
from funk_svd.dataset import fetch_ml_ratings
import argparse 

# Import PyTorch Ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.metrics import MeanSquaredError, MeanAbsoluteError


# Import Utility Functions
from loader import Loader
from datetime import datetime

# Import the Model Script
from MF import *


parser = argparse.ArgumentParser(description='toxcompl')
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--factors", type=int, default=15, help="factors")
parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
parser.add_argument("--bs", type=int, default=1024, help="batch size")
parser.add_argument("--r1", type=int, default=7, help="seed")
parser.add_argument("--r2", type=int, default=8, help="seed")
parser.add_argument("--path", type=str, default='./sample_input.csv', help="csv input")
parser.add_argument("--predfile", type=str, default='./predict.pkl', help="predictfile to write")
parser.add_argument("--full", type=int, default=1, help="whether use the full train set")
args = parser.parse_args()
print(args)

# Load preprocessed data

path = args.path
df = fetch_ml_ratings(path)
min_rating = df["rating"].min()
max_rating = df["rating"].max()

#n_user = len(df.u_id.unique())
#n_item = len(df.i_id.unique())

n_user = df.u_id.max()+1
n_item = df.i_id.max() + 1

print('n_user', n_user, 'n_item', n_item)

if args.full == 0:
    print('not using the full set to train')
    df = df.sample(frac=1).reset_index(drop=True) # shuffle the whole thing
    train = df.sample(frac=0.9, random_state=args.r1)
    val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=args.r2)
    test = df.drop(train.index.tolist()).drop(val.index.tolist())
else:
    print('using the full set to train')
    train = df
    val = df.sample(frac=0.1, random_state = args.r1)
    test = df.sample(frac=0.1, random_state = args.r2)
print(train)


# We have a bunch of feature columns and last column is the y-target
# Note Pytorch is finicky about need int64 types
train_x = train[['u_id','i_id']].astype(np.int32).to_numpy()
#train_x = train[['u_id','i_id']].to_numpy()
train_y = train['rating'].astype(np.float32).to_numpy()

# We've already split the data into train & test set
#test_x = test[['u_id','i_id']].astype(np.int64).to_numpy()
test_x = test[['u_id','i_id']].astype(np.int32).to_numpy()
test_y = test['rating'].astype(np.float32).to_numpy()

#HEREE ========================
# Extract the number of users and  items 
n_occu = 0

# Hyper-parameter settings
lr = args.lr  # Learning Rate
k = args.factors  # Number of dimensions per user, item

# Setup logging
log_dir = 'runs/mf_' + str(datetime.now()).replace(' ', '_')

# Instantiate the model class object
model = MF(n_user, n_item, k=k )

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model.to(device) 

# setup Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Create the ignite trainer
trainer = create_supervised_trainer(model, optimizer, model.loss, device = device)

# Use MSE MAE as metrics
metrics = {'evaluation': MeanSquaredError(), 'mae':MeanAbsoluteError()}

# Create ignite evaluator
evaluator = create_supervised_evaluator(model, metrics=metrics, device = device)

# Data loading using our Loader class
train_loader = Loader(train_x, train_y, batchsize=args.bs)
test_loader = Loader(test_x, test_y, batchsize=args.bs)


def log_training_loss(engine, log_interval=500):
    """
    Log the training loss
    """
    model.itr = engine.state.iteration  # Keep track of iterations
    if model.itr % log_interval == 0:
        fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"
        # log epochs and outputs
        msg = fmt.format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
        print(msg, flush=True)

trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)

def log_validation_results(engine):
    """
    Log the validation loss
    """
    # When triggered, run the validation set
    evaluator.run(test_loader)
    # Evaluation metrics
    avg_loss = evaluator.state.metrics['evaluation']
    mae = evaluator.state.metrics['mae']
    print("Epoch[{}] Validation MSE: {:.4f} MAE{:.4f}".format(engine.state.epoch, avg_loss, mae), flush=True)
    #writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)

trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

# Train for args.epochs epochs
trainer.run(train_loader, max_epochs=args.epochs)

# Save the model to target folder
model_path = './models/mf_side_feat_'+'factor_'+str(args.factors)+'_epochs_'+str(args.epochs)+'_full_'+str(args.full)+'.pth'
torch.save(model.state_dict(), model_path)
