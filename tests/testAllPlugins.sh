FILE_DIR=$(cd $(dirname $0); pwd)
ROOT_DIR=$(cd $FILE_DIR/../; pwd)

cd $ROOT_DIR
mkdir $ROOT_DIR/tests/plugins/model_temp -p
test_plugins_files=`ls $ROOT_DIR/tests/plugins | grep test`

for test_plugin_file in $test_plugins_files
do
    echo "================ Testing $test_plugin_file ================"
    python $ROOT_DIR/tests/plugins/$test_plugin_file
    if [ $? != 0 ]; then
        echo "Error occurs in running $test_plugin_file"
        exit $?
    fi
done