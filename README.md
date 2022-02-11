# nav-competition-icra2022

## Build Singularity image
```
sudo singularity build --notest nav_competition_image.sif Singularityfile.def
```

## Run navigation stack
```
./singularity_run.sh /path/to/image/file python3 run.py \
--world_idx 0 \
--navigation_stack jackal_helper/launch/DWA.launch
```