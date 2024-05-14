rm -rf triangulation_result

python3 packages/camera/scripts/triangulation_to_mcap.py \
        --mask-sources datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_3_1_mask datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_3_2_mask \
        --rgb-sources datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_3_1 datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_3_2 \
        --detection-result detection_result.json \
        --export triangulation_result \
        --intrinsic-params packages/camera/launch/params.yaml
