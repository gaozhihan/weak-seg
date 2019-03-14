test dataset is downloaded through http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar

'test.txt' originally locates in $DATA_DIR/VOC2012/VOCdevkit/VOC2012/ImageSets/

please put the file list 'test.txt' into your file list directory, e.g., $DATA_DIR/VOC2012_SEG_AUG/ImageSets/

run $ROOT_DIR/modified/test.py, the test script will save all 1456 test outputs to the dir specified by SaveTest.save_dir in save_test.py