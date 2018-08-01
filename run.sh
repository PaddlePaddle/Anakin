#for i in {1, 2, 4, 8, 16, 32, 64, 100}
for i in {10}
do 
../output/unit_test/test_map_rnn /home/chengyujuan/baidu/sys-hic-gpu/anakin-models/map/anakin-models/rnn/ gis_rnn.anakin2.bin 10000 1 0 $i
done
