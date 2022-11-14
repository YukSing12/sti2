import numpy as np
import os
data_Src="data\\perf.test.txt"

def loadLabelsAndData(path):
  assert os.path.exists(path),"文件不存在"
  with open(data_Src,"r") as fp:
      datalines=fp.readlines()
  data=[x.strip() for x in datalines]
  datasets=[]
  for i in data:
      temp_dict={"tensors":[]}
      label=i.split(";")
      temp_dict["qid"]=int(label[0][4:])
      try:
        temp_dict["label"]=int(label[1][6:])
      except:
        temp_dict["label"]=label[1][6:]
      for x in label[2:]:
        temp=[m.split() for m in x.split(":")]
        shape=tuple(int(k) for k in temp[0])
        value=[float(k) for k in temp[1]]
        temp_dict["tensors"].append(np.array(value).reshape(shape))
      datasets.append(temp_dict)
  return datasets

datasets=loadLabelsAndData(data_Src)
print(len(datasets))