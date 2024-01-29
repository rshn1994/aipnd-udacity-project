#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Roshan Sathyanarayana Shenoy      
# DATE CREATED: 03.11.2023                                  
# REVISED DATE: 04.11.2023
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir
import os
# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Replace None with the results_dic dictionary that you created with this
    # function
    results_list = listdir(image_dir)
    results_dic = {}
    pet_label_list = []
    
    for dog_file in results_list:
        pets_list = dog_file.lower().split("_")
        for i in range(len(pets_list)):
            if not pets_list[i].isalpha():
                del pets_list[i]
        pet_label =" ".join(pets_list)
        pet_label_list.append(pet_label)
    
    for i in range(len(results_list)):
        if ((results_list[i] not in results_dic) and (results_list[i] != ".")):
            results_dic[results_list[i]] = [pet_label_list[i]]
        else:
            print("** Warning: Duplicate files exist in directory:", in_files[idx])
            
    return results_dic

# if __name__ == "__main__":
#     from get_input_args import get_input_args
#     images = get_input_args().dir
#     labels = get_pet_labels(images)
#     print(labels)