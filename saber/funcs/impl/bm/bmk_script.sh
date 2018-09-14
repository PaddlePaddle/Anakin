ITCM_SZ_OCT=$(wc -c < bmkernel_bin_itcm.hex.sim)
ITCM_SZ=$(echo "obase=16;$ITCM_SZ_OCT" | bc)
echo 0x$ITCM_SZ
printf "%x" 0x$ITCM_SZ >> bmkernel_bin.bin