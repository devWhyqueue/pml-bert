import matplotlib.pyplot as plt
from datasets import load_dataset

train_files = [
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00000-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00001-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00002-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00003-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00004-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00005-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00006-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00007-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00008-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00009-of-00011.arrow',
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/train/data-00010-of-00011.arrow'
]

valid_files = [
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/validation/data-00000-of-00001.arrow'
]

test_files = [
    '/home/pml21/MS1/pml-bert/replibert/replibert/tokenized/civil_comments/test/data-00000-of-00001.arrow'
]

# load the tokenized datasets 
print("Loading datasets...")
train_dataset = load_dataset('arrow', data_files={'train': train_files}, split='train')
valid_dataset = load_dataset('arrow', data_files={'validation': valid_files}, split='validation')
test_dataset = load_dataset('arrow', data_files={'test': test_files}, split='test')
print("Finished loading datasets.")
def get_input_lengths(dataset):
    return [sum(mask) for mask in dataset['attention_mask']]

print("Generating input lengths...")
train_lengths = get_input_lengths(train_dataset)
valid_lengths = get_input_lengths(valid_dataset)
test_lengths = get_input_lengths(test_dataset)

print("Generating plot")
plt.figure(figsize=(10, 6))
plt.hist(train_lengths, bins=50, color='blue', alpha=0.6, label='Train')
plt.hist(valid_lengths, bins=50, color='green', alpha=0.6, label='Validation')
plt.hist(test_lengths, bins=50, color='red', alpha=0.6, label='Test')
plt.title("Distribution of Non-Padded Input ID Lengths")
plt.xlabel("Length of Input ID (Non-Padded)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('/home/pml21/MS1/pml-bert/replibert/replibert/plots/input_ids_distribution.png')
plt.close()
