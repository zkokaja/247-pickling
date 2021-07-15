import argparse
import os
import pickle
import string

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from transformers import (BartForConditionalGeneration, BartTokenizer,
                          BertForMaskedLM, BertTokenizer, GPT2LMHeadModel,
                          GPT2Tokenizer, RobertaForMaskedLM, RobertaTokenizer,
                          BlenderbotSmallTokenizer, BlenderbotTokenizer,
                          BlenderbotSmallForConditionalGeneration,
                          BlenderbotForConditionalGeneration)
from utils import create_folds, lcs, main_timer


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    add_ext = '' if file_name.endswith('.pkl') else '.pkl'

    file_name = file_name + add_ext
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def select_conversation(args, df):
    if args.conversation_id:
        df = df[df.conversation_id == args.conversation_id]
    return df


def load_pickle(args):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(args.pickle_name, 'rb') as fh:
        datum = pickle.load(fh)

    df = pd.DataFrame.from_dict(datum['labels'])

    return df


def add_glove_embeddings(df, dim=None):
    if dim == 50:
        glove = api.load('glove-wiki-gigaword-50')
        df['glove50_embeddings'] = df['token2word'].apply(
            lambda x: get_vector(x, glove))
    else:
        raise Exception("Incorrect glove dimension")

    return df


def check_token_is_root(args, df):
    if args.embedding_type == 'gpt2-xl':
        df['gpt2-xl_token_is_root'] = df['word'] == df['token'].apply(
            args.tokenizer.convert_tokens_to_string).str.strip()
    elif args.embedding_type == 'bert':
        df['bert_token_is_root'] = df['word'] == df['token']
    elif 'blenderbot' in args.embedding_type:
        df['bbot_token_is_root'] = df['word'] == df['token'].apply(
            args.tokenizer.convert_tokens_to_string).str.strip()
    else:
        raise Exception("embedding type doesn't exist")

    return df


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def convert_token_to_idx(df, tokenizer):
    df['token_id'] = df['token'].apply(tokenizer.convert_tokens_to_ids)
    return df


def tokenize_and_explode(args, df):
    """Tokenizes the words/labels and creates a row for each token

    Args:
        df (DataFrame): dataframe of labels
        tokenizer (tokenizer): from transformers

    Returns:
        DataFrame: a new dataframe object with the words tokenized
    """
    df['token'] = df.word.apply(args.tokenizer.tokenize)
    df = df.explode('token', ignore_index=True)

    # NOTE - this function doesn't work for blenderbot, act -> "a c t"
    df['token2word'] = df['token'].apply(
        args.tokenizer.convert_tokens_to_string).str.strip().str.lower()
    # df = remove_punctuation(df)
    df = convert_token_to_idx(df, args.tokenizer)
    # df = check_token_is_root(args, df)
    df = add_glove_embeddings(df, dim=50)

    # from tfspkl_main import add_vocab_columns
    # df = add_vocab_columns(df)

    return df


def get_token_indices(args, num_tokens):
    if args.embedding_type == 'gpt2-xl':
        start, stop = 0, num_tokens
    elif args.embedding_type == 'bert':
        start, stop = 1, num_tokens + 1
    else:
        raise Exception('wrong model')

    return (start, stop)


def map_embeddings_to_tokens(args, df, embed):

    multi = df.set_index(['conversation_id', 'sentence_idx', 'sentence'])
    unique_sentence_idx = multi.index.unique().values

    uniq_sentence_count = len(get_unique_sentences(df))
    assert uniq_sentence_count == len(embed)

    c = []
    for unique_idx, sentence_embedding in zip(unique_sentence_idx, embed):
        a = df['conversation_id'] == unique_idx[0]
        b = df['sentence_idx'] == unique_idx[1]
        num_tokens = sum(a & b)
        start, stop = get_token_indices(args, num_tokens)
        c.append(pd.Series(sentence_embedding[start:stop, :].tolist()))

    df['embeddings'] = pd.concat(c, ignore_index=True)
    return df


def get_unique_sentences(df):
    return df[['conversation_id', 'sentence_idx',
               'sentence']].drop_duplicates()['sentence'].tolist()


def process_extracted_embeddings(concat_output):
    """(batch_size, max_len, embedding_size)"""
    # concatenate all batches
    concatenated_embeddings = torch.cat(concat_output, dim=0).numpy()
    emb_dim = concatenated_embeddings.shape[-1]

    # the first token is always empty
    init_token_embedding = np.empty((1, emb_dim)) * np.nan

    extracted_embeddings = np.concatenate(
        [init_token_embedding, concatenated_embeddings], axis=0)

    return extracted_embeddings


def process_extracted_logits(args, concat_logits, sentence_token_ids):
    """Get the probability for the _correct_ word"""
    # (batch_size, max_len, vocab_size)

    # concatenate all batches
    prediction_scores = torch.cat(concat_logits, axis=1).squeeze()
    true_y = torch.tensor(sentence_token_ids).unsqueeze(-1)

    # if prediction_scores.shape[0] == 0:
    #     return [None], [None], [None]
    # elif prediction_scores.shape[0] == 1:
    #     true_y = torch.tensor(sentence_token_ids[0][1:]).unsqueeze(-1)
    # else:
    #     sti = torch.tensor(sentence_token_ids)
    #     true_y = torch.cat([sti[0, 1:], sti[1:, -1]]).unsqueeze(-1)

    prediction_probabilities = F.softmax(prediction_scores, dim=1)

    logp = np.log2(prediction_probabilities)
    entropy = [None] + torch.sum(-prediction_probabilities * logp,
                                 dim=1).tolist()

    top1_probabilities, top1_probabilities_idx = prediction_probabilities.max(
        dim=1)
    predicted_tokens = args.tokenizer.convert_ids_to_tokens(
        top1_probabilities_idx)
    predicted_words = [
        args.tokenizer.convert_tokens_to_string([token])
        for token in predicted_tokens
    ]

    # top-1 probabilities
    top1_probabilities = [None] + top1_probabilities.tolist()
    # top-1 word
    top1_words = [None] + predicted_words
    # probability of correct word
    true_y_probability = [None] + prediction_probabilities.gather(
        1, true_y).squeeze(-1).tolist()
    # TODO: probabilities of all words

    return top1_words, top1_probabilities, true_y_probability, entropy


def extract_select_vectors(batch_idx, array):
    if batch_idx == 0:
        x = array[0, :-1, :].clone()
        if array.shape[0] > 1:
            rem_sentences_preds = array[1:, -2, :].clone()
            x = torch.cat([x, rem_sentences_preds], axis=0)
    else:
        x = array[:, -2, :].clone()

    return x


def model_forward_pass(args, data_dl, hidden_states_kw='hidden_states'):
    model = args.model
    device = args.device
    # tokenizer = args.tokenizer

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        accuracy, count = 0, 0

        all_embeddings = []
        all_logits = []
        for batch_idx, batch in enumerate(data_dl):
            # batch = batch.to(args.device)
            input_ids = torch.LongTensor(batch['encoder_ids']).to(device)
            decoder_ids = torch.LongTensor(batch['decoder_ids']).to(device)
            model_output = model(input_ids.unsqueeze(0),
                                 decoder_input_ids=decoder_ids.unsqueeze(0))

            embeddings = model_output[hidden_states_kw][-1].cpu()[:, :-1, :]
            logits = model_output.logits.cpu()[:, :-1, :]

            predictions = model_output.logits.cpu().numpy().argmax(axis=-1)
            y_true = decoder_ids[1:].cpu().numpy()
            y_pred = predictions[0, :-1]
            accuracy += np.sum(y_true == y_pred)
            count += y_pred.size

            # Uncomment to debug
            # if batch_idx == 23 or batch_idx == 49:
            #     print(tokenizer.decode(batch['encoder_ids']))
            #     print(tokenizer.convert_ids_to_tokens(batch['decoder_ids'][1:]))
            #     print(tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1).squeeze().tolist()))
            #     print()
            #     breakpoint()

            # embeddings = extract_select_vectors(batch_idx, embeddings)
            # logits = extract_select_vectors(batch_idx, logits)

            all_embeddings.append(embeddings)
            all_logits.append(logits)

    print('model_forward accuracy', accuracy / count)
    return all_embeddings, all_logits


def get_conversation_tokens(df, conversation):
    token_list = df[df.conversation_id == conversation]['token_id'].tolist()
    return token_list


def make_input_from_tokens(args, token_list):
    size = args.context_length

    # HG's approach
    if len(token_list) <= size:
        windows = [tuple(token_list)]
    else:
        windows = [
            tuple(token_list[x:x + size])
            for x in range(len(token_list) - size + 1)
        ]

    # ZZ's approach
    # windows = [
    #     tuple(token_list[max(i - size, 0):i])
    #     for i in range(1,
    #                    len(token_list) + 1)
    # ]
    return windows


def make_dataloader_from_input(windows):
    input_ids = torch.tensor(windows)
    data_dl = data.DataLoader(input_ids, batch_size=2, shuffle=False)
    return data_dl


def make_conversational_input(args, df):
    '''
    Create a conversational context/response pair to be fed into an encoder
    decoder transformer architecture. The context is a seires of utterances
    that precede a new utterance response.
    '''

    examples = []
    utterances = [row.iloc[-1].sentence for _, row in df.groupby('sentence_idx')]
    # convo = args.tokenizer(list(utterances))['input_ids']

    bos = args.tokenizer.bos_token_id
    eos = args.tokenizer.eos_token_id
    sep = args.tokenizer.sep_token_id

    sep_id = [sep] if sep is not None else [eos]
    bos_id = [bos] if bos is not None else [sep]
    convo = [bos_id + row.token_id.values.tolist() + sep_id for _, row in df.groupby('sentence_idx')]
    # convo_special = []
    # for conv in convo:
    #   convo_special.append(bos_id + conv + sep_id)
    # convo = convo_special

    def create_context(conv, last_position, max_tokens=128):
        ctx = []
        for p in range(last_position, 0, -1):
            if len(ctx) + len(conv[p]) > max_tokens:
                break
            ctx = conv[p] + ctx
        return ctx

    for j, response in enumerate(convo):
        # Skip first n responses b/c of little context?
        if j == 0:
            continue

        # Add
        context = create_context(convo, j-1)
        if len(context) > 0:
            examples.append({
                'encoder_ids': context,
                'decoder_ids': response[:-1]
                })

    return examples


def printe(example, args):
    tokenizer = args.tokenizer
    print(tokenizer.decode(example['encoder_ids']))
    print(tokenizer.convert_ids_to_tokens(example['decoder_ids']))
    print()


def generate_conversational_embeddings(args, df):
    df = tokenize_and_explode(args, df)

    final_embeddings = []
    final_top1_word = []
    final_top1_prob = []
    final_true_y_prob = []
    for conversation in df.conversation_id.unique():
        # token_list = get_conversation_tokens(df, conversation)
        # model_input = make_input_from_tokens(args, token_list)
        # input_dl = make_dataloader_from_input(model_input)
        # input_dl = data.DataLoader(examples, batch_size=2, shuffle=False)
        df_convo = df[df.conversation_id == conversation]
        examples = make_conversational_input(args, df_convo)
        # print(sum([len(e['decoder_ids']) for e in examples]))
        input_dl = examples
        embeddings, logits = model_forward_pass(args, input_dl,
                                                hidden_states_kw='decoder_hidden_states')

        # embeddings = process_extracted_embeddings(embeddings)
        # assert embeddings.shape[0] == len(token_list)

        # NOTE - temporary hack.
        diff = len(df) - sum(e.shape[1] for e in embeddings)
        emb_dim = embeddings[0].shape[-1]
        other = [torch.full((1, diff, emb_dim), float('nan'))]
        embeddings = other + embeddings

        embeddings = torch.cat(embeddings, dim=1).numpy().squeeze()
        final_embeddings.append(embeddings)
        # assert len(embeddings) == len(df_convo)

        y_true = np.concatenate([e['decoder_ids'][1:] for e in input_dl])
        top1_word, top1_prob, true_y_prob, entropy = process_extracted_logits(
            args, logits, y_true)

        # NOTE - temporary hack
        y_pres = [None]*(diff - 1)
        final_top1_word.extend(y_pres)
        final_top1_prob.extend(y_pres)
        final_true_y_prob.extend(y_pres)

        final_top1_word.extend(top1_word)
        final_top1_prob.extend(top1_prob)
        final_true_y_prob.extend(true_y_prob)

    # print(len(final_top1_word), embeddings.shape, df.shape)
    df['embeddings'] = np.concatenate(final_embeddings, axis=0).tolist()
    df['top1_pred'] = final_top1_word
    df['top1_pred_prob'] = final_top1_prob
    df['true_pred_prob'] = final_true_y_prob
    df['surprise'] = -df['true_pred_prob'] * np.log2(df['true_pred_prob'])
    # df['entropy'] = entropy  # wth

    print('ZZ2 Accuracy', (df.token == df.top1_pred).mean())

    return df


def generate_embeddings_with_context(args, df):
    df = tokenize_and_explode(args, df)
    # df[['word', 'token', 'gpt2-xl_token_is_root']].to_csv('new_df.csv')
    if args.embedding_type == 'gpt2-xl':
        args.tokenizer.pad_token = args.tokenizer.bos_token

    final_embeddings = []
    final_top1_word = []
    final_top1_prob = []
    final_true_y_prob = []
    for conversation in df.conversation_id.unique():
        token_list = get_conversation_tokens(df, conversation)
        model_input = make_input_from_tokens(args, token_list)
        print(len(model_input))
        # print(model_input)
        input_dl = make_dataloader_from_input(model_input)
        embeddings, logits = model_forward_pass(args, input_dl)
        print(len(embeddings), len(logits))
        embeddings = process_extracted_embeddings(embeddings)
        assert embeddings.shape[0] == len(token_list)
        final_embeddings.append(embeddings)

        top1_word, top1_prob, true_y_prob, entropy = process_extracted_logits(
            args, logits, y_true)
        final_top1_word.extend(top1_word)
        final_top1_prob.extend(top1_prob)
        final_true_y_prob.extend(true_y_prob)

    # TODO: convert embeddings dtype from object to float
    df['embeddings'] = np.concatenate(final_embeddings, axis=0).tolist()
    df['top1_pred'] = final_top1_word
    df['top1_pred_prob'] = final_top1_prob
    df['true_pred_prob'] = final_true_y_prob
    df['surprise'] = -df['true_pred_prob'] * np.log2(df['true_pred_prob'])
    df['entropy'] = entropy

    return df


def generate_embeddings(args, df):
    tokenizer = args.tokenizer
    model = args.model
    device = args.device

    model = model.to(device)
    model.eval()

    df = tokenize_and_explode(args, df)
    unique_sentence_list = get_unique_sentences(df)

    if args.embedding_type == 'gpt2-xl':
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(unique_sentence_list, padding=True, return_tensors='pt')
    input_ids_val = tokens['input_ids']
    attention_masks_val = tokens['attention_mask']

    dataset = data.TensorDataset(input_ids_val, attention_masks_val)
    data_dl = data.DataLoader(dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        concat_output = []
        for batch in data_dl:
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
            }
            model_output = model(**inputs)
            concat_output.append(model_output[-1][-1].detach().cpu().numpy())

    embeddings = np.concatenate(concat_output, axis=0)
    emb_df = map_embeddings_to_tokens(args, df, embeddings)

    return emb_df


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def generate_glove_embeddings(args, df):
    glove = api.load('glove-wiki-gigaword-50')
    df['embeddings'] = df['word'].apply(lambda x: get_vector(x, glove))

    return df


def setup_environ(args):

    DATA_DIR = os.path.join(os.getcwd(), 'data', args.project_id)
    RESULTS_DIR = os.path.join(os.getcwd(), 'results', args.project_id)
    PKL_DIR = os.path.join(RESULTS_DIR, args.subject, 'pickles')

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    labels_file = '_'.join([args.subject, args.pkl_identifier, 'labels.pkl'])
    args.pickle_name = os.path.join(PKL_DIR, labels_file)

    args.input_dir = os.path.join(DATA_DIR, args.subject)
    args.conversation_list = sorted(os.listdir(args.input_dir))

    args.gpus = torch.cuda.device_count()
    if args.gpus > 1:
        args.model = nn.DataParallel(args.model)

    stra = '_'.join([args.embedding_type, 'cnxt', str(args.context_length)])

    # TODO: if multiple conversations are specified in input
    if args.conversation_id:
        args.output_dir = os.path.join(RESULTS_DIR, args.subject, 'embeddings',
                                       stra, args.pkl_identifier)
        os.makedirs(args.output_dir, exist_ok=True)

        output_file_name = args.conversation_list[args.conversation_id - 1]
        args.output_file = os.path.join(args.output_dir,
                                        output_file_name)

        args.output_file_prefinal = os.path.join(
            args.output_dir, output_file_name + '_prefinal')

    return


def select_tokenizer_and_model(args):

    if args.embedding_type == 'gpt2-xl':
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2LMHeadModel
        model_name = 'gpt2-xl'
    elif args.embedding_type == 'roberta':
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForMaskedLM
        model_name = 'roberta'
    elif args.embedding_type == 'bert':
        tokenizer_class = BertTokenizer
        model_class = BertForMaskedLM
        model_name = 'bert-large-uncased-whole-word-masking'
    elif args.embedding_type == 'bart':
        tokenizer_class = BartTokenizer
        model_class = BartForConditionalGeneration
        model_name = 'bart'
    elif args.embedding_type == 'blenderbot-small':
        tokenizer_class = BlenderbotSmallTokenizer
        model_class = BlenderbotSmallForConditionalGeneration
        model_name = 'facebook/blenderbot_small-90M'
    elif args.embedding_type == 'blenderbot':
        tokenizer_class = BlenderbotTokenizer
        model_class = BlenderbotForConditionalGeneration
        model_name = 'facebook/blenderbot-3B'  # NOTE
        model_name = 'facebook/blenderbot-400M-distill'  # NOTE
    elif args.embedding_type == 'glove50':
        return
    else:
        print('No model found for', args.model_name)
        exit(1)

    # CACHE_DIR = None
    CACHE_DIR = os.path.join(os.path.dirname(os.getcwd()), '.cache')
    os.makedirs(CACHE_DIR, exist_ok=True)

    # TODO add_prefix_space=True,
    args.tokenizer = tokenizer_class.from_pretrained(model_name,
                                                     cache_dir=CACHE_DIR)
    args.model = model_class.from_pretrained(model_name,
                                             cache_dir=CACHE_DIR,
                                             output_hidden_states=True)

    if args.history and args.context_length <= 0:
        args.context_length = args.tokenizer.max_len_single_sentence
        assert args.context_length <= args.tokenizer.max_len_single_sentence, \
            'given length is greater than max length'

    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        default='bert-large-uncased-whole-word-masking')
    parser.add_argument('--embedding-type', type=str, default='glove')
    parser.add_argument('--context-length', type=int, default=0)
    parser.add_argument('--save-predictions',
                        action='store_true',
                        default=False)
    parser.add_argument('--save-hidden-states',
                        action='store_true',
                        default=False)
    parser.add_argument('--subject', type=str, default='625')
    parser.add_argument('--history', action='store_true', default=False)
    parser.add_argument('--conversational', action='store_true', default=False)
    parser.add_argument('--conversation-id', type=int, default=0)
    parser.add_argument('--pkl-identifier', type=str, default=None)
    parser.add_argument('--project-id', type=str, default=None)

    args = parser.parse_args()
    if 'blenderbot' in args.embedding_type:
        args.conversational = True

    # custom_args = [
    #     '--project-id', 'podcast', '--pkl-identifier', 'full',
    #     '--conversation-id', '1', '--subject', '661', '--history',
    #     '--context-length', '1024', '--embedding-type', 'gpt2-xl'
    # ]

    return args


def tokenize_transcript(file_name):
    # Read all words and tokenize them
    with open(file_name, 'r') as fp:
        data = fp.readlines()

    data = [(i, item, item.strip().split(' ')) for i, item in enumerate(data)]
    data = [(item, sent, i) for i, sent, sublist in data for item in sublist]

    return data


def tokenize_podcast_transcript(args):
    """Tokenize the podcast transcript and return as dataframe

    Args:
        args (Namespace): namespace object containing project parameters
                            (command line arguments and others)

    Returns:
        DataFrame: containing tokenized transcript
    """
    DATA_DIR = os.path.join(os.getcwd(), 'data', args.project_id)
    story_file = os.path.join(DATA_DIR, 'podcast-transcription.txt')
    # story_file = os.path.join(DATA_DIR, 'pieman_transcript.txt')

    data = tokenize_transcript(story_file)

    df = pd.DataFrame(data, columns=['word', 'sentence', 'sentence_idx'])
    df['conversation_id'] = 1

    return df


def align_podcast_tokens(args, df):
    """Align the embeddings tokens with datum (containing onset/offset)

    Args:
        args (Namespace): namespace object containing project parameters
        df (DataFrame): embeddings dataframe

    Returns:
        df (DataFrame): aligned/filtered dataframe (goes into encoding)
    """
    DATA_DIR = os.path.join(os.getcwd(), 'data', args.project_id)
    cloze_file = os.path.join(DATA_DIR, 'podcast-datum-cloze.csv')
    # cloze_file = os.path.join(DATA_DIR, 'piemanAligned_all.txt')

    cloze_df = pd.read_csv(cloze_file, sep=',')
    words = list(map(str.lower, cloze_df.word.tolist()))

    # model_tokens = df['token2word'].tolist()
    model_tokens = df['token'].tolist()

    # Align the two lists
    mask1, mask2 = lcs(words, model_tokens)

    cloze_df = cloze_df.iloc[mask1, :].reset_index(drop=True)
    df = df.iloc[mask2, :].reset_index(drop=True)

    df_final = pd.concat([df, cloze_df], axis=1)
    df = df_final.loc[:, ~df_final.columns.duplicated()]

    return df


@main_timer
def main():
    args = parse_arguments()
    select_tokenizer_and_model(args)
    setup_environ(args)
    print(f'Reading the pickle {args.pickle_name}')

    if args.project_id == 'tfs':
        utterance_df = load_pickle(args)
        utterance_df = select_conversation(args, utterance_df)  # was commented
    elif args.project_id == 'podcast':
        labels_df = load_pickle(args)
        utterance_df = tokenize_podcast_transcript(args)
    else:
        raise Exception('Invalid Project ID')

    # NOTE - temporary
    if True:  # use already generated embeddings (e.g. to update labels)
        print('Using pregenerated embeddings')

        # Load pickle with embeddings
        with open(args.output_file + '.pkl', 'rb') as f:
            ds = pickle.load(f)
        df = pd.DataFrame(ds)

        # Align new labels with old embeddings
        labels_df['word'] = labels_df.word.replace("its'", "its")  # NOTE
        mask1, mask2 = lcs(labels_df.word.str.lower().tolist(),
                           df.token2word.tolist())
        print(len(mask2), df.shape)

        # NOTE - this only adds new columns, doesn't replace old ones
        labels_df2 = labels_df.iloc[mask1].copy()
        for col in set(df.columns).difference(set(labels_df.columns)):
            labels_df2[col] = df[col].values

        save_pickle(labels_df2.to_dict('records'), args.output_file)
        exit()

    if args.history:
        if args.embedding_type == 'gpt2-xl':
            df = generate_embeddings_with_context(args, utterance_df)
    elif args.conversational:
        df = generate_conversational_embeddings(args, utterance_df)
    else:
        if args.embedding_type == 'glove50':
            df = generate_glove_embeddings(args, utterance_df)
        else:
            df = generate_embeddings(args, utterance_df)

    if args.project_id == 'podcast':
        save_pickle(df.to_dict('records'), args.output_file_prefinal)
        df = align_podcast_tokens(args, df)
        # Folds should be created later
        # df = create_folds(df, 10)

    print(f'Saving to {args.output_file}')
    save_pickle(df.to_dict('records'), args.output_file)

    return


if __name__ == '__main__':
    main()
