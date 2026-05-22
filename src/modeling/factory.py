from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

def create_model(config_file): 
    '''
    Gets the appropriate model and tokenizer information from the .yaml file in the configs file

    Args: 
        config_file: The .yaml file in the configs file containing information about the model name 

    Returns: 
        The model and tokenizer associated with the model name
    '''
    
    # Obtain model name from the inputted config file
    model_name = config_file['model_information']['model_name']

    # Run a loop to return the appropriate model and tokenizer associated with the model name 
    if model_name == 'ChemBERTa-77M-MTR': 
        tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        model = RobertaForSequenceClassification.from_pretrained("DeepChem/ChemBERTa-77M-MTR"
                                                         ,num_labels=2
                                                         ,use_safetensors=True)
    elif model_name == 'MolFormer-XL':
        model = AutoModelForSequenceClassification.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                          deterministic_eval=True, 
                                          trust_remote_code=True, 
                                          num_labels=2, 
                                          use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", 
                                                  trust_remote_code=True)
    elif model_name == 'BARTSmiles': 
        model = AutoModelForSequenceClassification.from_pretrained("gayane/BARTSmiles", 
                                                                   num_labels=2, 
                                                                   use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained("gayane/BARTSmiles", 
                                                  add_prefix_space=True) 
    elif model_name == 'SELFIES-TED':
        tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")
        model = AutoModelForSequenceClassification.from_pretrained("ibm/materials.selfies-ted", 
                                                                   num_labels=2) 
    elif model_name == 'Mol-Gen': 
        tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")
        model = AutoModelForSequenceClassification.from_pretrained("zjunlp/MolGen-large", 
                                                      num_labels=2, 
                                                      use_safetensors=True)
        
    return model, tokenizer