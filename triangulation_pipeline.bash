python3 packages/camera/scripts/mask_to_centroid.py \
        --source datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_1_1_mask datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_1_2_mask \
        --export centroid_from_mask

make build-all

cd install/camera/lib/camera/

./triangulation ../../share/camera/launch/params.yaml /handy/centroid_from_mask_1.json /handy/centroid_from_mask_2.json new_detections.json

mv new_detections.json /handy

cd /handy

rm -rf triangulation_result

python3 packages/camera/scripts/triangulation_to_mcap.py \
        --mask-sources datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_1_1_mask datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_1_2_mask \
        --rgb-sources datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_1_1 datasets/TableOrange2msRecord_22_04/orange_dark_2ms/orange_dark_2ms_1_2 \
        --detection-result new_detections.json \
        --export triangulation_result \
        --intrinsic-params packages/camera/launch/params.yaml \
        # --transform-cam-to-world
