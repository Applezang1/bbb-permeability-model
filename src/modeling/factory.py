from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers import AutoTokenizer

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
        return model, tokenizer