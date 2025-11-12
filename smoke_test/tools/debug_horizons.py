#!/usr/bin/env python
import sys
from astropy.time import Time
from astroquery.jplhorizons import Horizons

def try_query(label, id_value, location, epochs):
    print(f"\n-- {label} --")
    obj = Horizons(id=id_value, id_type=None, location=location, epochs=epochs)
    try:
        payload = obj.vectors(get_query_payload=True)
        uri = getattr(obj, 'uri', None)
        print("payload keys:", list(payload.keys()))
        # Show COMMAND/CENTER for clarity
        print("COMMAND=", payload.get('COMMAND'), "CENTER=", payload.get('CENTER'))
        if 'TLIST' in payload:
            tlist_raw = payload['TLIST']
            # Coerce to list[float]
            if isinstance(tlist_raw, str):
                # Split on commas/whitespace and filter empties
                parts = [p for p in tlist_raw.replace("\n", ",").replace(" ", ",").split(',') if p]
                tlist = [float(p) for p in parts]
            elif isinstance(tlist_raw, (list, tuple)):
                tlist = [float(v) for v in tlist_raw]
            else:
                try:
                    import numpy as np
                    tlist = [float(v) for v in np.array(tlist_raw).ravel().tolist()]
                except Exception:
                    tlist = []
            if tlist:
                print(f"TLIST len={len(tlist)} min={min(tlist):.6f} max={max(tlist):.6f}")
            else:
                print("TLIST present but could not parse values")
        else:
            print("RANGE:", payload.get('START_TIME'), payload.get('STOP_TIME'), payload.get('STEP_SIZE') or payload.get('STEP'))
        if uri:
            print("uri head:", uri[:200], "...")
        tab = obj.vectors()
        print("OK rows:", len(tab))
        print("first row jd=", float(tab['datetime_jd'][0]))
        print("x,y,z (AU)=", float(tab['x'][0]), float(tab['y'][0]), float(tab['z'][0]))
    except Exception as e:
        print("ERROR:", type(e).__name__, e)

def main():
    # Earliest test epoch: 2026-10-31 12:04:09.998 TDB (as seen in validator)
    t = Time(61344.502894, format='mjd', scale='tdb')
    jd = float(t.tdb.jd)
    print("Test epoch JD_TDB=", jd, "ISO_TDB=", t.tdb.iso)

    try_query("single epoch @0", "-211", "@0", jd)
    try_query("three epochs @0", "-211", "@0", [jd, jd+0.5, jd+1.0])
    # Also try Sun-centered just to probe server behavior
    try_query("single epoch @sun", "-211", "@10", jd)

if __name__ == "__main__":
    main()
