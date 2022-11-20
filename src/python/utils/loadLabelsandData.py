import numpy as np
import os

def loadLabelsAndData(path):
  assert os.path.exists(path),"文件不存在"
  with open(path,"r") as fp:
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
        # content=np.zeros(())
        value=np.array(value).reshape(shape)
        if shape[1] !=128 and shape[1] !=1:
          value=np.pad(value,((0,0),(0,128-shape[1]),(0,0)),"constant",constant_values=0)
        temp_dict["tensors"].append(np.ascontiguousarray(value))
      temp_dict["batch_size"]=shape[0]
      datasets.append(temp_dict)
  return datasets

if __name__ =="__main__":
    data_Src="/workspace/cgc/sti2/data/perf.test.txt"
    datasets=loadLabelsAndData(data_Src)
    print(len(datasets))