#!/bin/sh
# */AIPND-revision/intropyproject-classify-pet-images/run_models_batch.sh
#                                                                             
# PROGRAMMER: Jennifer S.
# DATE CREATED: 08.02.2018                                 
# REVISED DATE: 06.11.2023  - Roshan Sathyanarayana Shenoy
# PURPOSE: Runs all three models to test which provides 'best' solution.
#          Please note output from each run has been piped into a text file.
#
# Usage: sh run_models_batch.sh    -- will run program from commandline within Project Workspace
#  
python check_images.py --dir /home/workspace/pet_images --arch resnet  --dogfile /home/workspace/dognames.txt > resnet_pet-images_roshan.txt
python check_images.py --dir /home/workspace/pet_images --arch alexnet --dogfile /home/workspace/dognames.txt > alexnet_pet-images_roshan.txt
python check_images.py --dir /home/workspace/pet_images --arch vgg  --dogfile /home/workspace/dognames.txt > vgg_pet-images_roshan.txt
