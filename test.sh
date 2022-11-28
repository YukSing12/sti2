#rm *.res.txt
if [ $# != 1 ]; then
	echo "Enter <main/multiprofile>"
	exit 1
fi
./bin/$1 ./model/Ernie.plan ./data/label.test.txt ./label.res.txt ./so/plugins
./bin/$1 ./model/Ernie.plan ./data/perf.test.txt ./perf.res.txt ./so/plugins
python src/python/utils/local_evaluate.py ./label.res.txt
python src/python/utils/local_evaluate.py ./perf.res.txt
nan_num=`cat label.res.txt | grep nan | wc -l`
echo "Found $nan_num nan in label.res.txt"
nan_num=`cat perf.res.txt | grep nan | wc -l`
echo "Found $nan_num nan in perf.res.txt"

/usr/bin/python3 <<-EOF
import numpy as np
with open("data/perf.res.txt", 'r') as fid1,\
        open("perf.res.txt", 'r') as fid2:
    lines1 = fid1.readlines()
    lines2 = fid2.readlines()
    assert (len(lines1) == len(lines2))
    mean_diff = 0
    max_diff = 0
    count = 0
    for i in range(len(lines1)):
        line1 = lines1[i]
        line2 = lines2[i]
        data1 = np.array([float(x) for x in line1.split("\t")[2].split(",")])
        data2 = np.array([float(x) for x in line2.split("\t")[2].split(",")])
        abs_diff = np.abs(data1 - data2)
        max_diff = max_diff if max_diff > abs_diff.max() else abs_diff.max()
        mean_diff = np.sum(abs_diff)
        count += len(abs_diff)
    
    mean_diff /= count
    print("Max diff is {}".format(max_diff))
    print("Mean diff is {}".format(mean_diff))
EOF