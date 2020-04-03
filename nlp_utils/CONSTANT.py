CHI_WORD_LENGTH = 2
MAX_CHAR_LENGTH = 5000
MAX_VOCAB_SIZE=20000
MAX_SEQ_LENGTH=5000
MAX_VALID_PERCLASS_SAMPLE=400
MAX_VALID_TOTAL=2000
MAX_SAMPLE_TRIAN=18000
MAX_TRAIN_PERCLASS_SAMPLE=800
EMBEDDING_DIM = 300
TOKENIZER_SAMPLE_STR = 5000
TOKENIZER_MAX_TOKEN = 5000 * 300
LENGTH_SAMPLE_STR = 300
CV_RATE = 0.3

MAX_TRAIN_PER_SAMPLE = 1000
MAX_TRAIN_TOTAL = 2000

def get_proper_data(num_class):
    max_per_train = MAX_TRAIN_PER_SAMPLE * num_class
    max_total = MAX_TRAIN_TOTAL
    max_last = min([max_per_train, max_total])
    return get_rounded_batch_size(max_last, num_class)

def get_rounded_batch_size(number, classes):
    return int((number // classes) * classes)

def get_valid_per_sample(classes):
    return min([MAX_VALID_TOTAL // classes, MAX_VALID_PERCLASS_SAMPLE])