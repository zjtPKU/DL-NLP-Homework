from datasets import load_dataset, DatasetDict, Dataset
import random
import json
def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str or list of str, the name(s) of the dataset(s)
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    if isinstance(dataset_name, list):
        return aggregate_datasets(dataset_name, sep_token)

    if dataset_name == 'restaurant_sup':
        return prepare_restaurant_dataset(sep_token, few_shot=False)

    elif dataset_name == 'laptop_sup':
        return prepare_laptop_dataset(sep_token, few_shot=False)

    elif dataset_name == 'acl_sup':
        return prepare_acl_dataset(few_shot=False)

    elif dataset_name == 'agnews_sup':
        return prepare_agnews_dataset(few_shot=False)

    elif dataset_name == 'restaurant_fs':
        return prepare_restaurant_dataset(sep_token, few_shot=True)

    elif dataset_name == 'laptop_fs':
        return prepare_laptop_dataset(sep_token, few_shot=True)

    elif dataset_name == 'acl_fs':
        return prepare_acl_dataset(few_shot=True)

    elif dataset_name == 'agnews_fs':
        return prepare_agnews_dataset(few_shot=True)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def prepare_restaurant_dataset(sep_token, few_shot,few_shot_size=256):
    # Load and process SemEval-2014 Task 4 restaurant data
    processed_train = {'text': [], 'labels': []}
    processed_test = {'text': [], 'labels': []}
    dataset_dir = "SemEval14-res"
    with open(f"{dataset_dir}/train.json", 'r') as f:
        train_data = json.load(f)
    with open(f"{dataset_dir}/test.json", 'r') as f:
        test_data = json.load(f)
    label_dict = {}
    # print(type(train_data))
    for key, value in list(train_data.items()):
        text = value['term'] + sep_token + value['sentence']
        label = value['polarity']   
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        label = label_dict[label]
        processed_train['text'].append(text)
        processed_train['labels'].append(label)
    
    if few_shot:
        # Randomly sample a subset of the data
        combined = list(zip(processed_train['text'], processed_train['labels']))
        random.shuffle(combined)  # Shuffle the data
        sampled_combined = combined[:few_shot_size]  # Take a few-shot size sample
        processed_train['text'], processed_train['labels'] = zip(*sampled_combined)
        
        
    for key, value in list(test_data.items()):
        text = value['term'] + sep_token + value['sentence']
        label = value['polarity']   
        label = label_dict[label]
        processed_test['text'].append(text)
        processed_test['labels'].append(label)
    # print(label_dict)
    return DatasetDict({
        'train': Dataset.from_dict(processed_train),
        'test': Dataset.from_dict(processed_test)
    })

def prepare_laptop_dataset(sep_token, few_shot,few_shot_size=256):
    # Load and process SemEval-2014 Task 4 laptop data
    processed_train = {'text': [], 'labels': []}
    processed_test = {'text': [], 'labels': []}
    dataset_dir = "SemEval14-laptop"
    label_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
    with open(f"{dataset_dir}/train.json", 'r') as f:
        train_data = json.load(f)
    # for key, value in list(train_data.items())[:5]:
    #     print(key, value)
    with open(f"{dataset_dir}/test.json", 'r') as f:
        test_data = json.load(f)

    # print(type(train_data))
    for key, value in list(train_data.items()):
        text = value['term'] + sep_token + value['sentence']
        label = value['polarity']   
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        label = label_dict[label]
        processed_train['text'].append(text)
        processed_train['labels'].append(label)
    if few_shot:
        # Randomly sample a subset of the data
        combined = list(zip(processed_train['text'], processed_train['labels']))
        random.shuffle(combined)  # Shuffle the data
        sampled_combined = combined[:few_shot_size]  # Take a few-shot size sample
        processed_train['text'], processed_train['labels'] = zip(*sampled_combined)
    for key, value in list(test_data.items()):
        text = value['term'] + sep_token + value['sentence']
        label = value['polarity']   
        label = label_dict[label]
        processed_test['text'].append(text)
        processed_test['labels'].append(label)
    # print(label_dict)
    return DatasetDict({
        'train': Dataset.from_dict(processed_train),
        'test': Dataset.from_dict(processed_test)
    })

def prepare_acl_dataset(few_shot):
    # Load and process ACL-ARC data
    dataset = load_dataset('acl_sup') 
    # print(dataset)  
    train_data, test_data = dataset['train'], dataset['test']
    # print(train_data)
    # print(test_data)
    if few_shot:
        train_data = select_few_shot_samples(train_data, seed=42)

    processed_train = {'text': [], 'labels': []}
    processed_test = {'text': [], 'labels': []}
    label_dict = {}
    for entry in train_data:
        text = entry['text']
        label = entry['label']
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        label = label_dict[label]
        processed_train['text'].append(text)
        processed_train['labels'].append(label)

    for entry in test_data:
        text = entry['text']
        label = entry['label']
        label = label_dict[label]
        processed_test['text'].append(text)
        processed_test['labels'].append(label)
    # print(label_dict)
    return DatasetDict({
        'train': Dataset.from_dict(processed_train),
        'test': Dataset.from_dict(processed_test)
    })

def prepare_agnews_dataset(few_shot):
    # Load and process AG News data
    dataset = load_dataset('ag_news', split='test')
    # print(type(dataset))
    # print(dataset)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=2022)

    if few_shot:
        train_test_split['train'] = select_few_shot_samples(train_test_split['train'], seed=42)

    processed_train = {'text': train_test_split['train']['text'],
                       'labels': train_test_split['train']['label']}

    processed_test = {'text': train_test_split['test']['text'],
                      'labels': train_test_split['test']['label']}

    return DatasetDict({
        'train': Dataset.from_dict(processed_train),
        'test': Dataset.from_dict(processed_test)
    })

def select_few_shot_samples(dataset, seed, num_samples=256):
    random.seed(seed)
    sampled_indices = random.sample(range(len(dataset)), num_samples)
    return dataset.select(sampled_indices)

def aggregate_datasets(dataset_names, sep_token):
    aggregated_dataset = {'train': {'text': [], 'labels': []}, 'test': {'text': [], 'labels': []}}
    label_offset = 0

    for name in dataset_names:
        dataset = get_dataset(name, sep_token)
        for split in ['train', 'test']:
            for text, label in zip(dataset[split]['text'], dataset[split]['labels']):
                aggregated_dataset[split]['text'].append(text)
                aggregated_dataset[split]['labels'].append(label + label_offset)

        label_offset += max(dataset['train']['labels']) + 1  # Update offset to avoid label overlap
        print(label_offset)
    return DatasetDict({
        'train': Dataset.from_dict(aggregated_dataset['train']),
        'test': Dataset.from_dict(aggregated_dataset['test'])
})
