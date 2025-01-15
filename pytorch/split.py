import splitfolders

# dataset directory
dir = 'dataset'

# split dataset into train , test and val directory in 70 : 15 : 15 ratio
splitfolders.ratio(dir,"splitdataset",ratio=(0.7,0.15, 0.15))
