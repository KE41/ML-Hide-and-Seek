# Humanoid Hide & Seek Python Machine Learning Project

Two physics-based bipedal humanoids play hide-and-seek in a custom PyBullet arena with obstacles to hide behind.

- **Seeker:** uses tabular Q-learning to choose which direction to walk.
- **Hider:** uses potential fields to steer behind cover and away from the Seeker.

Both agents walk on the same underlying locomotion controller, which uses either a CPG or IK gait plus PD joint control.

**Result:** across 500 rounds, the Hider survived 72% of the time. The Seeker's sudden direction changes made the biped unstable and forced it to slow down, which gave the Hider time to reach cover.

**Next steps:** replace the Q-table with PPO, then train both agents together using only raycast observations.

## Run

```bash
pip install pybullet numpy
python HideSeekEnv.py
```

### Poster Explaining:
[Humanoid_Hideseek_PDF_Slides.pdf](https://github.com/user-attachments/files/32571651/Humanoid_Hideseek_PDF_Slides.pdf)

### Official Project Report:
[OfficialReport.pdf](https://github.com/user-attachments/files/32571664/OfficialReport.pdf)
