# Script to download project dataset from onedrive link. 

#!/bin/bash
wget -O output https://mcgill-my.sharepoint.com/:u:/g/personal/raghav_mehta_mail_mcgill_ca/EVEvhY9_jyVEk2uSZ8wZhFYBQ58C57I7ZB55jBocKwB5Jg?download=1
mv output dataset.zip
unzip dataset.zip
