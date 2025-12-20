# Skills

Set the following env vars:

OPENAI_API_KEY
BOSDYN_CLIENT_USERNAME
BOSDYN_CLIENT_PASSWORD
RAY_CONNECT

## Files
- [README.md](README.md)
  - This file
- [arm_grasp.py](arm_grasp.py)
  - Skill for walking to item, grasping item, and lifting item up
  - For manual testing with opencv input run, `python -m src.skills.spot.arm_grasp 192.168.80.3`
- [walk_to_object.py](walk_to_object.py)
  - Skill for walking to item
  - For manual testing with opencv input run, `python -m src.skills.spot.walk_to_object 192.168.80.3`
- [push_bar_door.py](push_bar_door.py)
  - Skill for push doors with push bars using the wrist of the arm
  - For manual testing with opencv input run, ` python -m src.skills.spot.push_bar_door 192.168.80.3`
- [grasp_door.py](grasp_door.py)
  - Skill for pull doors with a door handle that needs to be grasped to open
  - For manual testing with opencv input run, ` python -m src.skills.spot.grasp_door 192.168.80.3`
- [walk_to_landmark.py](walk_to_landmark.py)
  - Skill for walking to landmark in mapped graphnav environment
  - Requires map (unzip `src\robots\spot\graph_nav_maps\exp_floor_1.zip`) and annotations (see `src\config\mappers\graphnav\exp_floor_1.yaml`)
  - Run in front of localization fidicual
  - `python -m src.skills.spot.walk_to_landmark 192.168.80.3 -u 'src\robots\spot\graph_nav_maps\exp_floor_1\downloaded_graph' --upload-annotations src\config\mappers\graphnav\exp_floor_1.yaml`

### Automated skills - No humans input required/parameter selection by models

- [automated_skills/door.py](automated_skills/door.py)
  - Opens push bash doors or grasp handle pull doors
  - Make sure robot is in front of a door (push bar or pull grasp handle) and run
  - Requires OPENAI_API_KEY to be set in env var
  - `python -m src.skills.spot.automated_skills.door 192.168.80.3 --ray-server "ray://grail-machine-ip.neu.edu:10001"`
- [automated_skills/grasp.py](automated_skills/grasp.py)
  - Skill for grasping object by object name. -t forces top down grasp. Check code for other input options
  - Make sure robot is in front of a object
  - `python -m src.skills.spot.automated_skills.grasp 192.168.80.3 -t --ray-server "ray://grail-machine-ip.neu.edu:10001" --object-name "blue brick"`
- [automated_skills/walk.py](automated_skills/walk.py)
  - Skill for walking to objects in video of the robot. -d takes in distance to keep from object
  - Make sure robot is in front of a object
  - `python -m src.skills.spot.automated_skills.walk 192.168.80.3 -d 1.0 --ray-server "ray://grail-machine-ip.neu.edu:10001" --object-name "trashcan"`


### CLI
- `python -m src.skills.spot.CLI.main 192.168.80.3 -u 'src\robots\spot\graph_nav_maps\exp_floor_1\downloaded_graph' --upload-annotations src\config\mappers\graphnav\exp_floor_1.yaml`


### Helper Utils

- [automated_skills/image.py](automated_skills/image.py)
  - Util for merging two images with correct rotation and getting click point on camera

# TODO
- TODO: documentation for other skills
- TODO: Move Automated CLI code to main branch