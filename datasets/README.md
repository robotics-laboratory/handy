# Information about datasets

## TableMulticolorRecord_19_04

|  |  |
| --- | --- |
| Date | 19.04.2024 |
| Location | Table tennis club HSE |
| Setup | 2 cameras across the one side of the table, ~90 degress axis intersection |
| Calibration available | No |
| Annotation | \<ball-color>\_<light/dark>\_\<exposure> |

### Available sets
- 6ms exposure
    - With additional light (from camera ID=2)
        - green
        - orange
        - white
    - No additional light
        - green
        - orange
        - white


## TableOrange2msRecord_22_04

|  |  |
| --- | --- |
| Date | 22.04.2024 |
| Location | Table tennis club HSE |
| Setup | 2 cameras across the one side of the table, ~90 degress axis intersection |
| Calibration available | Yes (intrinsics + stereo) |
| Annotation | \<ball-color>\_<light/dark>\_\<exposure> |

### Available sets
- 2ms exposure
    - With additional light (from camera ID=2)
        - orange
    - No additional light
        - orange
- calibration boards
    - aruco board (GridBoard, 5x7 cells, 60mm marker, 30mm gap, DICT_5X5_250)
    - charuco board (CharucoBoard, 7x10 cells, 60mm cell, 40mm marker, DICT_5X5_250)
