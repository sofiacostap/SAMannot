# Full image correspondence check

Status: verified_image_correspondence
FRAMES images: 3802; sequentially decoded video images: 3814; header: 3814.
Match counts: {'unique_exact': 3802}

Intervals below contain contiguous uniquely matched FRAMES indices. Positive offset means video index = FRAMES index + offset.

| FRAMES interval | Video interval | Offset |
|---|---|---:|
| 0–2 | 0–2 | 0 |
| 3–3801 | 15–3813 | 12 |

Video indices without an identical FRAMES image: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
Unresolved FRAMES indices: []

This verifies image correspondence only. It neither cleans GT identities nor certifies old SAM PNG filename provenance.
