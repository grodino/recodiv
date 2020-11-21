# RecDiv : analysis of the diversity of musical algorithmic recommendations

## Structure
The entry point of the code is the `main.py` function at the root of the project. Calling this 
function will recreate all the figures shown in the report.

* The core logic (machine learning models, graph objects and methods) is located in `./recodiv/`. 
* The data folder (not hosted on Github) must be `./recodiv/data/{data_set_name}/`
* The automation logic (users-songs-tags graphs instanciation, ML model instanciation, figure 
computation... ) is located in `./automation/`. 

This project uses spotify's automation framework [luigi](https://github.com/spotify/luigi/)

## Licences
* The code in `recodiv/triversity/` is adapted from [triversity](https://github.com/Nobody35/triversity)
(mainly comments and formating modifications)

## Useful material
* [Tutorial](https://jessesw.com/Rec-System/) on collaborative filtering recsys
* [Implicit](https://implicit.readthedocs.io/en/latest/) collaborative filtering for implicit 
dataset library