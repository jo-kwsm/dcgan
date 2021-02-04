python utils/make_csv_file.py
python utils/make_configs.py --model DCGAN SAGAN --max_epoch 100 200 400 800 1000

files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python train.py "${filepath}/config.yaml"
        python evaluate.py "${filepath}/config.yaml"
    fi
done
