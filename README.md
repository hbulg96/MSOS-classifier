# MSOS-classifier
Audio classification system using heatmap based feature extraction system, based on work in high level sound taxonomies in the 'Making Sense of Sounds' research project

To use the system, open the 'master_classification' file in an IDE, and edit the 'path' variable to your directory for the making sense of sounds project 'development' or 'evaluation' audio sets.

The variable 'n' dictates which of the 5 categories will be included in classification, to loop through all 5, set "for n in range(5)"
The variable 'x' dictates how many files from the 'n' categories will be classified by the system.
Set x to "for n in range(100)" for the 'evaluation' dataset, or "for n in range(300)" for the 'development' dataset.
