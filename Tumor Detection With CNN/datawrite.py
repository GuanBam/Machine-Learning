import csv,os,cv2
def convert_img_to_csv(img_dir):
    with open(r"data.csv","a",newline="") as f:
        column_name=["label"]
        for i in range(100*100):
            column_name.extend(["pixel%d"%(i+1)])
        writer = csv.writer(f)
        writer.writerow(column_name)
        
        # modify the "yes" to "no" to add data from "no" folder
        img_temp_dir = r"data/resize/yes"
        img_list=os.listdir(img_temp_dir)
        for img_name in img_list:
            if not os.path.isdir(img_name):
                img_path = os.path.join(img_temp_dir,img_name)
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                row_data=["yes"]
                print("row",len(img))
                print("column",len(img[0]))
                if len(img)<100:                
                    up=(100-len(img))//2
                    down=100-len(img)-up
                    for i in range(up*100):
                        row_data.append(0)
                    if len(img[0])<100:
                        left=(100-len(img[0]))//2
                        right=100-len(img[0])-left
                        for k in img:
                            for j in range(left):
                                row_data.append(0)
                            row_data.extend(k)
                            for j in range(right):
                                row_data.append(0)
                    else:
                        for k in img:
                            row_data.extend(k)
                    for i in range(down*100):
                        row_data.append(0)
                else:
                    if len(img[0])<100:
                        left=(100-len(img[0]))//2
                        right=100-len(img[0])-left
                        for k in img:
                            for j in range(left):
                                row_data.append(0)
                            row_data.extend(k)
                            for j in range(right):
                                row_data.append(0)
                    else:
                        
                        row_data.extend(img.flatten())
                writer.writerow(row_data)
                print("##############")

if __name__== "__main__":
    convert_img_to_csv(r"data\resize")
