from utilities import *
from train import checkpoint

model, class_to_idx = load_checkpoint(args.checkpoint_path)

if args.device and torch.cuda.is_available():
    device = 'cuda:0'
elif args.device and torch.cuda.is_available() == False:
    print("gpu selected but gpu resources are not available, training on cpu")
    device = 'cpu'
elif args.device == False and torch.cuda.is_available() == False:
    device = 'cpu'

probs, classes = predict(args.image_path, model, args.topk, class_to_idx)

print ('Classes: ', classes)
print('Probability: ', probs)
