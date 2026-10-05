# XMLPO = Explainable Machine Learning Pipeline Ontology

This repo contains relevant files from the Explainable Machine Learning Pipeline Ontology (XMLPO) by Donika Xhani (2023) and its extension by Dimitri Micha von Benckendorff (2026).

## Citation

XMLPO was published in this paper:

Xhani, D., Rebelo Moreira, J. L., Van Sinderen, M., & Ferreira Pires, L. (2024). XMLPO: An Ontology for Explainable Machine Learning Pipeline. In C. Trojahn, D. Porello, & P. P. F. Barcelos (Eds.), <em>Frontiers in Artificial Intelligence and Applications</em>. IOS Press. https://doi.org/10.3233/FAIA241306

## Contents

The DonikaXhani2023 folder contains Donika's conceptual model in a .vpp file as well as the ontology in a .ttl file.

The DimitriMichaVonBenckendorff2026 folder contains Dimitri's extended conceptual model in a xmlpo-co12.vpp file, the extended ontology in a xmlpo-co12.ttl file, and the extended ontology with metadata in a xmlpo-co12.rdf file. Additionally, this folder contains turtle files of the updated xmlpo, operational ontology for the healthcare use case, and instances reflecting the findings from the problem investigation in this study. Furthermore, the use case specific results are saved in six .json files and transformed to triples for the use case specific knowledge graph saved as the can-xai-kg.ttl file. These transformations are presented in the explaining-CAN-screening.ipynb file. Finally, the aforementioned knowledge graph was completed with all classes, properties, and instances from the relevant turtle files saved as the can-xai-kg-all.ttl file.

The SvenVanDerPeet2026 folder contains Sven's internship report with a proof of concept for cardiac autonomic neuropathy screening automation in a .pdf file. The machine learning pipeline developed in this work is also provided as the main_ML_V2.py file that uses data in the GE-71_Data_Summary_Table_final.csv file.

## Contacts
This space is administered by:

Dimitri Micha von Benckendorff d.m.vonbenckendorff@student.utwente.nl (GitHub: researcherdimitrivb)

Donika Xhani d.xhani@utwente.nl (GitHub: DonikaXhani)

João Luiz Rebelo Moreira j.luizrebelomoreira@utwente.nl (GitHub: jonimoreira)
