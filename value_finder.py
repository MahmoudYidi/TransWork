import json
import csv
import numpy as np

i = 2

##### Copying from JSON to CSV ######
if i == 1:
    jsonFile = open('/home/mahmoud/Downloads/detr-main/mydata/annotations/New_Train.json', 'r')
    values = json.load(jsonFile)
    jsonFile.close()
    f = open("filemap.csv","a")
    cr = csv.writer(f,delimiter=',')

    for i in range(835):
        idValue = values['images'][i]['id']
        print(idValue)
        fileName = values['images'][i]['file_name']
        print(fileName)
        rows = [idValue, fileName]
        cr.writerow(rows) 


##### REARRANGING CONTENT OF CSV
if i == 2:
    data = []
    with open('filemap.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    #print(data)

    f = open("New.csv","a")
    cr = csv.writer(f,delimiter=',')
    for x in range(len(data)):
        #print(data[x][0])
        for j in range(len(data)):
            #print(data[i][1])
            if data[j][9] == str(data[x][1]):

                Rows = data[j][2:9]
                #print(Rows)
                cr.writerow(Rows)






######################### CHECKING IF CORRECT #####################

if i == 3:
    data = []
    with open('filemap.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
    #print(data)

    data2 = []
    with open('final_obtaine.csv', newline='') as csvfile:
        data2 = list(csv.reader(csvfile))
    #print(data2)

    check =[]
    for x in range(len(data)):
        #print(data[x][0])
        check.append(data[x][1])
    
    #print(check[834])
    #print(data2[int(check[834])][0])
    for y in range(len(check)):
        print(y)
        if data2[int(check[y])][0] == data[y][2]:
          print('True')



