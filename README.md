## Automated Detection and Assessment of Vertebral Fractures in CT images
This repository contains the code developed during a Master Thesis internship at DIAG by Michel Botros supervised by Nikolas Lessmann. 
Two algorithms were developed for:

### (1) Genant Classification (`genant_classifier/`)
This algorithm identifies vertebral compression fractures and predicts fractures grades following the commonly used Genant classification.
* Source code for development, training and validation (`genant_classifier/devel/`)
* Processor for applying the method to unseen images (`genant_classifier/processor/`)

### (2) Vertebral Abnormality Scoring (`shape_prediction/`)
This algorithm compares the shape of the vertebral body with the expected shape to calculate a Vertebral Abnormality Score. 
This score expresses how abnormal the vertebra looks and can be used to identify fractures.
* Source code for development, training and validation (`shape_prediction/devel/`)
* Processor for applying the method to unseen images (`shape_prediction/processor/`)


