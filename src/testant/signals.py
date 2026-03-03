"""Signal intersection helpers for fair receiver comparison.

When comparing antennas across receivers with different constellation support
(e.g. ZED-F9T has GLONASS, ZED-F9T-20B does not), analysis must restrict to
the signal intersection so that satellite count and C/N0 comparisons reflect
antenna performance, not receiver capability differences.
"""

# Maps human-readable constellation names (as used in receivers.toml) to the
# gnss_id values used in CSV data and throughout the codebase.
_CONSTELLATION_TO_GNSS_ID: dict[str, str] = {
    "GPS":     "GPS",
    "GLONASS": "GLO",
    "Galileo": "GAL",
    "BeiDou":  "BDS",
    "QZSS":    "QZSS",
    "SBAS":    "SBAS",
}

_GNSS_ID_TO_CONSTELLATION: dict[str, str] = {
    v: k for k, v in _CONSTELLATION_TO_GNSS_ID.items()
}


def load_receiver_signals(receivers_cfg: dict) -> dict[str, set[str]]:
    """Return {receiver_name: set_of_gnss_ids} from a parsed receivers.toml.

    If a receiver lacks ``available_signals``, it is omitted from the result
    (and the intersection degrades gracefully).
    """
    result: dict[str, set[str]] = {}
    for rx_name, rx_cfg in receivers_cfg.get("receiver", {}).items():
        constellations = rx_cfg.get("available_signals")
        if constellations is None:
            continue
        gnss_ids: set[str] = set()
        for c in constellations:
            gid = _CONSTELLATION_TO_GNSS_ID.get(c)
            if gid:
                gnss_ids.add(gid)
        result[rx_name] = gnss_ids
    return result


def signal_intersection(receiver_signals: dict[str, set[str]]) -> set[str]:
    """Compute the gnss_id intersection across all receivers."""
    if not receiver_signals:
        return set()
    sets = list(receiver_signals.values())
    return sets[0].intersection(*sets[1:]) if len(sets) > 1 else sets[0]


def excluded_constellations(
    receiver_signals: dict[str, set[str]],
    intersection: set[str],
) -> dict[str, list[str]]:
    """Return {receiver_name: [excluded constellation names]} for reporting."""
    result: dict[str, list[str]] = {}
    all_gnss_ids: set[str] = set()
    for ids in receiver_signals.values():
        all_gnss_ids |= ids
    excluded = all_gnss_ids - intersection
    if excluded:
        for rx_name, gnss_ids in receiver_signals.items():
            only_here = gnss_ids & excluded
            if only_here:
                result[rx_name] = sorted(
                    _GNSS_ID_TO_CONSTELLATION.get(g, g) for g in only_here
                )
    return result


def exclusion_note(
    receiver_signals: dict[str, set[str]],
    intersection: set[str],
) -> str:
    """One-line note for plot subtitles describing what was filtered out."""
    excl = excluded_constellations(receiver_signals, intersection)
    if not excl:
        return ""
    parts = []
    for rx, consts in sorted(excl.items()):
        parts.append(f"{', '.join(consts)} ({rx} only)")
    return "Excluded: " + "; ".join(parts)
