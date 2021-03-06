# argoverse_lidar
This repository allows you to access to the pointcloud of the lidars in [argoverse dataset](https://www.argoverse.org/) separately.

Upper LiDAR | Lower LiDAR
------------ | -------------
![](https://github.com/irenecortes/argoverse_lidar/blob/master/images/up.png) | ![](https://github.com/irenecortes/argoverse_lidar/blob/master/images/down.png)

## Installation
It is needed to apply some changes to the [argoverse-api](https://github.com/argoai/argoverse-api):
- With diff.patch file
- Or manually:

*In /argoverse/data_loading/argoverse_tracking_loader.py:*

Modify the import of load_ply:
```diff
-      from argoverse.utils.ply_loader import load_ply
+      from argoverse.utils.ply_loader import load_ply, load_ply_ring
```

Add this function:

```python
def get_lidar_ring(self, idx: int, log_id: Optional[str] = None, load: bool = True) -> Union[str, np.ndarray]:
    """Get lidar corresponding to frame index idx (in lidar frame).

    Args:
        idx: Lidar frame index
        log_id: ID of log to search (default: current log)
        load: whether to load up the data, will return path to the lidar file if set to false

    Returns:
        Either path to lidar at a specific index, or point cloud data if load is set to True
    """
    assert self.lidar_timestamp_list is not None
    assert self._lidar_timestamp_list is not None
    assert self.lidar_list is not None
    assert self._lidar_list is not None

    if log_id is None:
        log_id = self.current_log

    assert idx < len(self._lidar_timestamp_list[log_id])

    if load:
        return load_ply_ring(self._lidar_list[log_id][idx])
    return self._lidar_list[log_id][idx]
```


*In /argoverse/utils/ply_loader.py:*

Add this function:
```python
def load_ply_ring(ply_fpath: _PathLike) -> np.ndarray:
    """Load a point cloud file from a filepath.

    Args:
        ply_fpath: Path to a PLY file

    Returns:
        arr: Array of shape (N, 3)5
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    i = np.array(data.points.intensity)[:, np.newaxis]
    ring = np.array(data.points.laser_number)[:, np.newaxis]

    return np.concatenate((x, y, z, i, ring), axis=1)
```
## Use
The script *get_velodyne.py* reads the lidar pointclouds from the argoverse dataset. You can configure which log to reproduce by changing this variables:
```python
    root_dir =  'ARGOVERSE_ROOT/argoverse-api/argoverse-tracking/sample/'
    log_id = 'c6911883-1843-3727-8eaa-41dc8cda8993'
```
The script uses ROS and publishes both pointclouds in different messages. 
The *velo2car.launch* file publishes the tf transformations between the lidars and the vehicle coordinate system.
