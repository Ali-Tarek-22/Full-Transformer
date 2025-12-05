from transformer import Transformer
import torch
import numpy as np

# Load the dataset from Huggingface
df = pd.read_csv("hf://datasets/salehalmansour/english-to-arabic-translate/en_ar_final.tsv", sep="\t")

# Get all rows that contain at least one null value
rows_with_null = df[df.isnull().any(axis=1)] # axis=1 to search through columns

print(f'Number of null rows = {len(rows_with_null)}\n')
print("Null rows:")
print(rows_with_null)
print("_" * 100)

print("Null count per column:")
print(df.isnull().sum())

# Nulls in specific columns
en_null = df[df['en'].isnull()]
ar_null = df[df['ar'].isnull()]

print("_" * 100)
print(f'Number of English nulls = {len(en_null)}')
print(f'Number of Arabic nulls = {len(ar_null)}')
print("_" * 100)


# Drop rows with any null values
df = df.dropna()

# values without Nulls
Arabic = df['ar']
English = df['en']
print(len(Arabic))
print(len(English))

# Vocabulary
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PADDING_TOKEN = '<PADDING>'

arabic_voc = [
     START_TOKEN, END_TOKEN, PADDING_TOKEN, ' ', '؟', '!', '.', ',',  #  punctuation
    '0','1','2','3','4','5','6','7','8','9',  # digits
    'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', # letter
    'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل',
    'م', 'ن', 'ه', 'و', 'ى', 'ي']

english_voc = [START_TOKEN, END_TOKEN, PADDING_TOKEN, ' ', '?', '!', '.', ',',  #  punctuation
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', # digits
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', # letter
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                        'y', 'z']

# character to index maps
arabic_to_index = {c:i for i, c in enumerate(arabic_voc)}
english_to_index = {c:i for i, c in enumerate(english_voc)}
index_to_arabic = {i:c for i, c in enumerate(arabic_voc)}
index_to_english = {i:c for i, c in enumerate(english_voc)}
print(arabic_to_index)
print(english_to_index)
print(index_to_arabic)
print(index_to_english)

# Convert vocab to string for regex (escape special characters)
# re.escape(c) → ensures special characters like ?, ., ! are treated literally in regex
arabic_chars = ''.join([re.escape(c) for c in arabic_voc])
english_chars = ''.join([re.escape(c) for c in english_voc])
print(arabic_chars)
print(english_chars)

# Optional Arabic normalization function
def normalize_arabic(text):
    text = str(text)
    # remove diacritics (tashkeel) if present
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    # normalize Alef variants
    text = text.replace('آ', 'ا').replace('أ', 'ا').replace('إ', 'ا')
    # normalize final Yaa
    text = text.replace('ى', 'ي')
    return text

# Remvoe characters from Arabic column not in Arabic vocabulary
# f"[^{arabic_chars}]" → matches any character NOT in vocabulary
# re.sub(..., "", x) → removes all characters not in the list
# Preprocessing Arabic
Arabic = Arabic.apply(lambda x: re.sub(f"[^{arabic_chars}]", "", normalize_arabic(x)))

# Preprocessing English
# Remvoe characters from English column not in English vocabulary
English = English.apply(lambda x: re.sub(f"[^{english_chars}]", "", str(x).lower()))

# English vodab
print(f'English vocabulary: {english_voc}')
print(f'English vocab size = {len(english_voc)}')
# English to index map
print(f'English to index map: {english_to_index}')
# Index to English map
print(f'Index to English map: {index_to_english}')
print("_"*10)
# Arabic vodab
print(f'Arabic vocabulary {arabic_voc}')
print(f'Arabic vocab size = {len(arabic_voc)}')
# Arabic to index map
print(f'Arabic to index map: {arabic_to_index}')
# Index to Arabic map
print(f'Index to Arabic map: {index_to_arabic}')
print("_"*10)

# random_idx = np.random.randint(0, len(English))
# print(f'Random English sentence:\n{English.iloc[random_idx]}')
# print(f'It\'s Arabic Translation:\n{Arabic.iloc[random_idx]}')

# Convert inot lists
english = English.tolist()
arabic = Arabic.tolist()

print(f'Number of examples: {len(english)}')

random_idx = np.random.randint(0, len(English))
print(f'Random English sentence:\n{english[random_idx]}')
print(f'It\'s Arabic Translation:\n{arabic[random_idx]}')

# Set the max sentece length to be more than the length of 98% of th data
english_98th_len  = np.percentile([ len(sentence) for sentence in english], 98)
arabic_98th_len = np.percentile([ len(sentence) for sentence in arabic], 98)
max_sequence_length  = min(english_98th_len, arabic_98th_len)


# check of the sentence characters all in the vocabulary
def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

# check if sentnece is shorter than the max sentence length
def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length)


valid_sentence_indicies = []
for index in range(len(arabic)):
    arabic_sentence, english_sentence = arabic[index], english[index]
    if is_valid_length(arabic_sentence, max_sequence_length) \
      and is_valid_length(english_sentence, max_sequence_length) \
      and is_valid_tokens(arabic_sentence, arabic_voc) and is_valid_tokens(english_sentence, english_vocab):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(arabic)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

# Only continue with valid sentences
arabic_sentences = [arabic[i] for i in valid_sentence_indicies]
english_sentences = [english[i] for i in valid_sentence_indicies]

# Set a mximum number of examples
TOTAL_SENTENCES = 1000000
arabic_sentences = arabic_sentences[:TOTAL_SENTENCES]
english_sentences = english_sentences[:TOTAL_SENTENCES]

class TextDataset(Dataset):

    def __init__(self, english_sentences, arabic_sentences):
        self.english_sentences = english_sentences
        self.arabic_sentences = arabic_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.arabic_sentences[idx]


dataset = TextDataset(english_sentences, arabic_sentences)
# dataset[5:10]
print(len(dataset))

batch_size = 128
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

print("First 3 batches:")
for batch_num, batch in enumerate(iterator):
  if batch_num>2:
    break
  batch_eng, batch_ar = batch
  print(batch_eng)
  print(batch_ar)

# split the dataset into train and test sets
train_size = int(0.9 * len(dataset))   # 90% train
test_size  = len(dataset) - train_size # 10% test

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f'Size of training set: {len(train_dataset)}')
print(f'Size of test set: {len(test_dataset)}')


d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 80
ar_vocab_size = len(arabic_voc)
batch_size = 128
num_epochs = 10

transformer = Transformer(d_model,
                          ffn_hidden,
                          num_heads,
                          drop_prob,
                          num_layers,
                          max_sequence_length,
                          ar_vocab_size,
                          english_to_index,
                          arabic_to_index,
                          START_TOKEN,
                          END_TOKEN,
                          PADDING_TOKEN)

transformer

def train_transformer(transformer, dataset, batch_size=128, num_epochs, lr=1e-4, device=None):
    # check which device is available
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transformer.to(device)

    # Load the training dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    PADDING_TOKEN = transformer.encoder.sentence_embedding.PADDING_TOKEN
    token_to_index = transformer.encoder.sentence_embedding.language_to_index

    # Loss function ignoring padding
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_index[PADDING_TOKEN], reduction = 'none') # none to get the loss of each sentence, set it to mean to get the mean loss of the batch (a single scaler)

    # Optimizer
    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)

    # Xavier initialization for weights > 1D
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    for epoch in range(num_epochs):
        print(f"______________________________ Epoch {epoch + 1}/{num_epochs} ______________________________")
        transformer.train()

        for batch_num, batch in enumerate(train_loader):
            # Unpack batch and ensure each sentence is a list
            src_sentences, tgt_sentences = batch
            src_sentences = list(src_sentences)
            tgt_sentences = list(tgt_sentences)

            optimizer.zero_grad()

            # Forward pass
            predictions = transformer(
                src_sentences,
                tgt_sentences,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True
            )

            # Tokenize target sentences for loss (the ground truth)
            labels = transformer.decoder.sentence_embedding.batch_tokenize(
                tgt_sentences, start_token=False, end_token=True
            ).to(device)

            # Compute loss
            loss = criterion(predictions.view(-1, predictions.size(-1)), labels.view(-1))
            # Compute mean loss without the padding tokens
            valid_indicies = torch.where(labels.view(-1) == token_to_index[PADDING_TOKEN], False, True)
            # Compute average loss through the batch
            loss = loss.sum() / valid_indicies.sum()
            # Or also compute the mean using:
            # loss_value = loss.sum() / (labels.view(-1) != padding_idx).sum()

            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()

            if (batch_num + 1) % 100 == 0 or (batch_num + 1) == len(train_loader):
                print(f"Epoch {epoch + 1}, Batch {batch_num + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

train_transformer(transformer, train_dataset, batch_size, num_epochs)

# Save parameters
save_dir = "/content/transformer.pt"
torch.save(transformer.state_dict(), save_dir)

# Load parameters to same architecture transformer
checkpoint = torch.load(save_dir, map_location="cuda")
transformer.load_state_dict(checkpoint)


def test_transformer(transformer, test_dataset, batch_size=128, device=None):
    if dNevice is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transformer.to(device)

    # Load test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get padding token index
    PADDING_TOKEN = transformer.encoder.sentence_embedding.PADDING_TOKEN
    token_to_index = transformer.encoder.sentence_embedding.language_to_index
    padding_idx = token_to_index[PADDING_TOKEN]

    # Same loss setup as training (ignoring padding)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='none')

    total_loss = 0
    total_batches = 0

    transformer.eval()
    with torch.no_grad():
        for batch_num, batch in enumerate(test_loader):
            src_sentences, tgt_sentences = batch
            src_sentences = list(src_sentences)
            tgt_sentences = list(tgt_sentences)

            # Forward pass
            predictions = transformer(
                src_sentences,
                tgt_sentences,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=True
            )

            # Tokenize the true sentences
            labels = transformer.decoder.sentence_embedding.batch_tokenize(
                tgt_sentences, start_token=False, end_token=True
            ).to(device)

            # Compute per-token loss
            loss = criterion(predictions.view(-1, predictions.size(-1)),
                             labels.view(-1))

            # Mask out padding and compute mean
            valid = labels.view(-1) != padding_idx
            loss = loss.sum() / valid.sum()

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"\nTest Loss: {avg_loss:.4f}")

    return avg_loss


def translate(ip_sentence, device=None):
  if device is None:
      device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

  transformer.eval()
  transformer.to(device) # Ensure the model is on the correct device

  ip_sentence = (ip_sentence.lower(),)
  op_sentence = ("",)

  max_sequence_length = transformer.decoder.sentence_embedding.max_sequence_length
  # lookup table
  idx_to_token = transformer.decoder.sentence_embedding.language_to_index

  END_TOKEN = transformer.decoder.sentence_embedding.END_TOKEN
  END_IDX = idx_to_token[END_TOKEN]


  for step in range(max_sequence_length):
    predictions = transformer(
            ip_sentence,
            op_sentence,
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,   # important
            dec_end_token=False     # we don't want end token in interference
        )

    next_token_prob_distribution = predictions[0][step]
    next_token_index = torch.argmax(next_token_prob_distribution).item() # .item() to convert a 0-dim PyTorch tensor into a plain Python number.
    next_token = idx_to_token[next_token_index]

    if next_token == END_TOKEN:
      break

    op_sentence = (op_sentence[0] + next_token, )

  return  op_sentence[0]

sentence = translate(transformer, "how are you")
print(sentence)

sentence = translate(transformer, "I went there today")
print(sentence)

sentence = translate(transformer, "show me your strength")

print(sentence)
