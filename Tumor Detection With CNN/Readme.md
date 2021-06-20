Dataset from Kaggle. 
246 Exapmples in total, 155 labeled as Yes, 91 labeled as No.

## Step 1 Resize Figure
Since the original figure are in different size, I resized them first with "resize.py" into size 100*100

## Step 2 Turn the Figure Into CSV
Still some figure may smaller than 100*100, when turn them into CSV, I will fill the edge with Black to make sure the size will be 100*100

## Step 3 Training and Test
Tensorflow is used for the training model
