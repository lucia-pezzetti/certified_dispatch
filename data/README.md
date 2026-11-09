# Road network data

Both files are derivative databases of [OpenStreetMap](https://www.openstreetmap.org)
data, available under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
© OpenStreetMap contributors.

| File | Nodes | Directed links | Used by |
|---|---:|---:|---|
| `manhattan_south.json` | 2,087 | 4,250 | All Manhattan experiments of the paper (Section IV-B and IV-C) |
| `manhattan_small.json` | 72 | 123 | Unit tests only |

Each node keeps its OpenStreetMap identifier and longitude/latitude. Link directions
are preserved and reverse links are never invented. Geometry is reduced to node
coordinates and straight connecting segments. Both graphs are strongly connected,
so every sampled service can be completed. The small file is the largest strongly
connected component of the crop of the large one to longitude [-74.014, -74.005] and
latitude [40.705, 40.713].

Every link carries a nominal free-flow time `free_flow_minutes` and a bounded delay
width `delay_width_minutes`. The traversal time of a link in the simulator is

```text
traversal_time = free_flow_minutes + Uniform(0, delay_width_minutes)
```

drawn independently at every traversal. Delays and demand are synthetic: the nominal
times are model inputs, not measured journey times, and the snapshot date of the
OpenStreetMap extract was not recorded, so the data describe the topology of the area,
not a current road survey. The JSON files also store the provenance fields of the
extraction (parent file hash, crop, selection rule, noise definition, attribution).

Every run copies the network it used into its output directory as `network.json`
and records its SHA-256 in `metadata.json`.
