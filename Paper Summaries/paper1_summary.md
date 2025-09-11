# Paper 1: A curated diverse molecular database of blood-brain barrier permeability with chemical descriptors 

<ins>Paper 1 Link</ins>: https://www.nature.com/articles/s41597-021-01069-5 

## Key Terminology for Contextual Understanding: 

<ins>SMILES</ins>: representation of chemical structure as a line of text and symbols

<ins>Pandas DataFrames</ins>: a 2-dimensional data structure where each column can hold different types of data (integer, string, float) 

<ins>PubChem CID</ins>: a unique number that is associated with a unique chemical compound stored in the PubChem Compound Database 

<ins>RdKit</ins>: an open-source cheminformatics library that allows you to find information related to molecules 

<ins>Isomeric SMILES</ins>: a type of SMILES that represents bonds between different elements as well as their stereochemical and isotopic information

<ins>Canonical SMILES</ins>:  a type of SMILES that represents bonds between different elements

<ins>stereochemical</ins>: the 3D arrangement of atoms

<ins>InChI (International Chemical Identifier)</ins>: a unique representation of chemical structure as a line of text and symbols that is longer than SMILES but contains stereochemical and isometric information

<ins>logBB threshold value</ins>: a determinant value that determines whether a drug is permeable (above threshold) or not permeable to the BBB (below threshold)

<ins>IUPAC name</ins>: a readable representation of a certain chemical molecule intended to be used as the vocabulary for scientific works

## Summary
# Steps for Obtaining and Revising Data: 
<ins>Reference Papers</ins>: Find articles and papers containing BBB permeability and/or logBB information for each molecular compound

<ins>Extract Data from File</ins>:

    - If the document is a PDF: 

        - Use tabula-py to convert the PDF data into a list of pandas DataFrames, which can be stored in Excel XLSX 

        - Convert data into float64 to ensure numerical operations

    - DOCX, DOC, CSV, TXT… any Excel-compatible format: 

        - Convert to Excel XLSX directly using Microsoft Office

<ins>Revising and Cleaning Data</ins>:

    - Fix invalid SMILES: Remove white spaces or line breaks from the SMILES string 
    to create a valid SMILES string 

    - Add PubChem CID: Molecules that are missing identifiers, such as the SMILES string, 
    PubChem CID, and compound name, can be found by inputting given information to 
    PubChemPy, a public NIH database with chemical information.

    - Other Potential Issues:

        - Problem: Compound name is marked with multiple PubChem instances, resulting 
        in ambiguity. 

            - Fix: Put a flag and mark for potential ambiguity 

        - Problem: Only the molecular structure is given. 

            - Fix: Use the PubChem interface to build the molecule and search for 
            the PubChem CID and SMILES string. 

        - Note: SMILES string checked for validity by loading into RdKit 

<ins>Add Isomeric/Canonical SMILES</ins>: 

    - Problem: Normal SMILES strings don’t have stereochemical information, which influences 
    the permeability of the specific compound through the BBB membrane 

    - Solution: Use PUG-REST API to get isomeric SMILES using normal SMILES, or use PubChemPy to get 
    canonical SMILES if isomeric isn’t available 

<ins>ChEMBL Structure Pipeline</ins>:

    - Run SMILES through the ChEMBL Structure Pipeline, which cleans up the SMILES string by:

        - Striping salts 

        - Neutralizing the charge 

        - Removing molecules containing metal atoms or heavy atoms with an atomic number greater than 20  

        - Updating valence electrons 

        - And more… 

<ins>Reupdate PubChem CID</ins>: 

    - ChEMBL Structure Pipeline outputs a canonical or isomeric SMILES that can be different from the original, so use PubChemPy 
    to update all the information again and store it.

<ins>Data Curation</ins> 

    - Problem: InChI is unique molecularly, but can’t resolve tautomeric form (same molecules but different structure)

    - Solution: Use a unique InChI (generated with RdKit) and isomeric/canonical SMILES (which distinguishes differences in 
    stereochemical information) for a unique chemical identifier. 
    
<ins>Data Curation</ins>: Filtering and Grouping Data

    - Remove molecules with logBB values that are outliers (based on the distribution of logBB values)  
    
    - Check molecules where their logBB value was reported multiple times in multiple papers. Remove molecules where the 
    reported data are significantly different from each other (Equation for Determining Significance: max(logBB) - min(logBB) > 1)

    - Grouping Molecular Compounds:

        - Group A: Molecules with one unique logBB value 

        - Group B: Molecules with more than one logBB value, where each reported logBB value has 
        a difference of less than 5% from the mean logBB value 

        - Group C: Molecules with two distinct logBB values 

        - Group D: Molecule with more than two distinct logBB values, the highest frequency logBB value 
        is chosen as the value to use  

        - Discard molecules that fail to be grouped 

    - Grouping Categorical Data: 

        - Group A: Molecules with numerical data with different logBB threshold values used for different papers 

        - Group B: Molecules whose BBB permeability is quantified by a threshold value of logBB = -1 and 
        have the same reported permeability result from multiple sources

        - Group C: Molecules with no threshold value for BBB permeability, but have the same reported
        permeability result from multiple sources 

        - Group D: Molecules with different reported permeability results. The most frequent permeability result 
        is chosen to represent the molecule, and any molecule with an equal number of both permeability results is discarded. 
