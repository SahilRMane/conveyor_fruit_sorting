# Project status

The simulated Panda tracks moving trays, inspects fruit cues and sorts trays into two bins.
The pipeline uses capture-time RGB images, MoveIt 2, Cartesian Servo and physical finger contact.
The supervisor keeps arm command ownership exclusive.

| Checkpoint | Recorded scope |
| --- | --- |
| M4 integration | 10 of 10 seeded cases passed at 1 mm/s, with direct sorts, inspection sorts and passes |
| M5 offline selector | 192 held-out cases per deadline window. In the widest window, the learned selector made 144 correct accepts and the fixed-left policy made 96. Both made zero wrong accepts. |
| M5 live inspection | Left, right and late cases passed with the frozen selector |
| M5 physical sorting | The left-view pilot sorted to bin A. The right-view pilot sorted to bin B. Both passed reset and ended in HOLD with zero command overlap. |
| M5 seeded selector | 11 of 11 seeded cases passed. Five trays reached the correct bins and six trays exited. Every reset passed with zero command overlaps. |

These results use simulated trays with red and green cues.
The current code includes a two-view selector, independent evaluation and NVIDIA camera rendering.
The source tree includes the benchmark code. Run reports and trained selector files are generated locally.
