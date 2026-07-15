from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartForSequenceClassification


def create_model(config_file, 
                 classifier_dropout): 
    '''
    Gets the appropriate model and tokenizer information from the .yaml file in the configs file

    Args: 
        config_file: The .yaml file in the configs file containing information about the model name 
        classifier_dropout: The probability of a neuron in the classifer layer being temporarily droppped during training

    Returns: 
        The model and tokenizer associated with the model name
    '''
    
    # Obtain model name from the inputted config file
    model_name = config_file['model_information']['model_name']

    # Run an if/else statement to return the appropriate model and tokenizer associated with the model name 
    if model_name == 'ChemBERTa-77M-MTR': 
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-77M-MTR"
                                                         ,num_labels=2
                                                         ,use_safetensors=True
                                                         ,classifier_dropout=classifier_dropout)
    elif model_name == 'MolFormer-XL':
        model = AutoModelForSequenceClassification.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                          deterministic_eval=True, 
                                          trust_remote_code=True, 
                                          num_labels=2, 
                                          use_safetensors=True, 
                                          classifier_dropout_prob=classifier_dropout)
        tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                                  trust_remote_code=True)
    elif model_name == 'BARTSmiles': 
        model = AutoModelForSequenceClassification.from_pretrained("gayane/BARTSmiles", 
                                                                   num_labels=2, 
                                                                   use_safetensors=True, 
                                                                   classifier_dropout=classifier_dropout)
        tokenizer = AutoTokenizer.from_pretrained("gayane/BARTSmiles", 
                                                  add_prefix_space=True) 
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<s>"
    elif model_name == 'SELFIES-TED':
        tokenizer = AutoTokenizer.from_pretrained("ibm-research/materials.selfies-ted")
        model = BartForSequenceClassification.from_pretrained("ibm-research/materials.selfies-ted", 
                                                                   num_labels=2, 
                                                                   classifier_dropout=classifier_dropout)
    elif model_name == 'Mol-Gen': 
        tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")
        model = AutoModelForSequenceClassification.from_pretrained("zjunlp/MolGen-large", 
                                                      num_labels=2, 
                                                      use_safetensors=True, 
                                                      classifier_dropout=classifier_dropout)
    else: 
        raise ValueError('Model name not recognized')

    return model, tokenizer


def dataset_loader(config_file): 
    '''
    Returns the appropriate molecule column name according to the dataset information
    from the .yaml file in the configs file

    Args: 
        config_file: The .yaml file in the configs file containing information about the dataset name 

    Returns: 
        The appropriate molecule column name
    '''
    
    # Obtain dataset name from the inputted config file
    dataset_name = config_file['dataset_information']['dataset']

    # Run an if/else statement to return the appropriate dataset loading function for the dataset
    if dataset_name == 'data/processed/smiles_dataframe.csv': 
        return 'SMILES'  
    elif dataset_name == 'data/processed/selfies_dataframe.csv':
        return 'SELFIES'
    else: 
        raise ValueError('Dataset path not recognized')