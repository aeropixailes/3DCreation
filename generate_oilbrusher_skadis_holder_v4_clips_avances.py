
import math
import numpy as np
import trimesh
from shapely.geometry import Point, box
from pathlib import Path

# ============================================================
# Support SKADIS Oilbrusher - V4
# Corrections :
# - 6 stylos Oilbrusher Ø16,5 mm
# - fond réel sous les stylos : tablette continue + fonds circulaires individuels
# - grande plaque arrière : les crochets SKADIS ne sont plus "dans le vide"
# - 4 crochets hauts + 1 crochet central bas
# - écart vertical entre ligne haute et crochet bas : 45 mm
# - crochets repris depuis le STL de référence, avec léger serrage
# - supports circulaires avancés légèrement pour rester bien ronds
# ============================================================

REF_STL_CANDIDATES = [
    "/mnt/data/skadis_tamiya_23ml(1).stl",
    "/mnt/data/skadis_tamiya_23ml.stl",
]

OUTPUT = "oilbrusher_skadis_holder_160mm_6clips_v4_clips_avances.stl"

# Géométrie générale
WIDTH = 160.0
BACKPLATE_H = 67.0
BACKPLATE_T = 3.0

# Porte-stylos
CLIP_COUNT = 6
PEN_DIAMETER = 16.5
CLEARANCE = 0.8
WALL = 2.0
OPENING = 13.0
CLIP_HEIGHT = 50.0
SIDE_MARGIN = 5.0
CLIP_FORWARD_OFFSET = 2.6  # avance légère des supports circulaires, sans modifier platine/crochets

# Fonds / tablette
SHELF_T = 3.0
BOTTOM_DISC_T = 2.6
BOTTOM_DISC_CLEARANCE = 0.3

# Crochets SKADIS
TOP_HOOK_X = [20.0, 60.0, 100.0, 140.0]
LOWER_HOOK_X = [80.0]
TOP_HOOK_Z_MIN = 52.0
LOWER_HOOK_Z_MIN = TOP_HOOK_Z_MIN - 45.0
HOOK_TIGHTEN_X = 1.035   # léger serrage latéral
HOOK_TIGHTEN_Z = 1.020   # léger serrage vertical

# Extraction du crochet depuis le modèle de référence
HOOK_OFFSETS_X = [-20.0, 20.0]  # on privilégie un crochet central propre
HOOK_HALF_WINDOW_X = 2.6
HOOK_Y_MIN = -16.5
HOOK_Z_MAX = 12.5


def find_reference_stl():
    for p in REF_STL_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError("Aucun STL de référence SKADIS trouvé dans /mnt/data")


def extract_reference_hook():
    """Extrait un crochet SKADIS du STL de référence et le normalise.

    Repère final du crochet :
    - X centré sur 0
    - Y = 0 sur la face d'accrochage côté plaque arrière
    - le crochet dépasse vers Y négatif
    - Z min = 0
    """
    ref_path = find_reference_stl()
    ref = trimesh.load(ref_path)
    cent = ref.triangles_center
    ref_center_x = (ref.bounds[0, 0] + ref.bounds[1, 0]) / 2.0

    best = None
    best_area = -1.0

    for off in HOOK_OFFSETS_X:
        cx = ref_center_x + off
        mask = (
            (cent[:, 0] > cx - HOOK_HALF_WINDOW_X)
            & (cent[:, 0] < cx + HOOK_HALF_WINDOW_X)
            & (cent[:, 1] > HOOK_Y_MIN)
            & (cent[:, 2] < HOOK_Z_MAX)
        )
        sub = ref.submesh([mask], append=True, repair=False)
        comps = sub.split(only_watertight=False)
        if not comps:
            continue
        comp = max(comps, key=lambda m: m.area)
        if comp.area > best_area:
            best = comp
            best_area = comp.area

    if best is None:
        raise RuntimeError("Impossible d'extraire un crochet depuis le STL de référence")

    hook = best.copy()
    b = hook.bounds
    x_center = (b[0, 0] + b[1, 0]) / 2.0
    y_attach = b[0, 1]   # dans le STL d'origine, c'est la face côté support
    z_min = b[0, 2]
    z_center = (b[0, 2] + b[1, 2]) / 2.0

    # X centré, Z démarré à 0, Y retourné pour dépasser vers l'arrière
    v = hook.vertices.copy()
    v[:, 0] = v[:, 0] - x_center
    v[:, 1] = -(v[:, 1] - y_attach)
    v[:, 2] = v[:, 2] - z_min

    # Léger serrage pour réduire le jeu dans SKADIS
    zc = z_center - z_min
    v[:, 0] *= HOOK_TIGHTEN_X
    v[:, 2] = (v[:, 2] - zc) * HOOK_TIGHTEN_Z + zc

    hook.vertices = v
    hook.merge_vertices()
    return hook


def place_hook(hook_ref, x, z_min):
    h = hook_ref.copy()
    h.apply_translation([x, 0.0, z_min])
    return h


def make_u_clip(cx, cy, z0, height):
    """Crée un clip vertical type U ouvert vers l'avant (+Y)."""
    inner_d = PEN_DIAMETER + CLEARANCE
    r_in = inner_d / 2.0
    r_out = r_in + WALL

    cut_offset = math.sqrt(max(r_in**2 - (OPENING / 2.0)**2, 0.0))

    outer = Point(cx, cy).buffer(r_out, resolution=96)
    inner = Point(cx, cy).buffer(r_in, resolution=96)

    # ouverture frontale, côté +Y
    front_cut = box(cx - 40, cy + cut_offset, cx + 40, cy + 40)
    profile = outer.difference(inner.union(front_cut))

    clip = trimesh.creation.extrude_polygon(profile, height=height, engine="earcut")
    clip.apply_translation([0.0, 0.0, z0])
    return clip


def make_bottom_disc(cx, cy):
    """Fond circulaire individuel à la base de chaque stylo."""
    inner_d = PEN_DIAMETER + CLEARANCE
    r_in = inner_d / 2.0
    radius = r_in - BOTTOM_DISC_CLEARANCE

    disc = trimesh.creation.cylinder(
        radius=radius,
        height=BOTTOM_DISC_T,
        sections=96
    )
    disc.apply_translation([cx, cy, BOTTOM_DISC_T / 2.0])
    return disc


def build_model():
    parts = []

    inner_d = PEN_DIAMETER + CLEARANCE
    r_in = inner_d / 2.0
    r_out = r_in + WALL

    # Grande plaque arrière, indispensable pour que les crochets ne soient pas en porte-à-faux
    backplate = trimesh.creation.box(extents=[WIDTH, BACKPLATE_T, BACKPLATE_H])
    backplate.apply_translation([WIDTH / 2.0, BACKPLATE_T / 2.0, BACKPLATE_H / 2.0])
    parts.append(backplate)

    # Tablette/fond continu sous tous les stylos
    cy = BACKPLATE_T + r_in + CLIP_FORWARD_OFFSET
    shelf_depth = cy + r_out + 2.0
    shelf = trimesh.creation.box(extents=[WIDTH, shelf_depth, SHELF_T])
    shelf.apply_translation([WIDTH / 2.0, shelf_depth / 2.0, SHELF_T / 2.0])
    parts.append(shelf)

    # Crochets SKADIS repris du STL de référence
    hook_ref = extract_reference_hook()

    for x in TOP_HOOK_X:
        parts.append(place_hook(hook_ref, x, TOP_HOOK_Z_MIN))

    for x in LOWER_HOOK_X:
        parts.append(place_hook(hook_ref, x, LOWER_HOOK_Z_MIN))

    # Positions des 6 porte-stylos
    x_left = SIDE_MARGIN + r_out
    x_right = WIDTH - SIDE_MARGIN - r_out
    xs = np.linspace(x_left, x_right, CLIP_COUNT)

    for x in xs:
        parts.append(make_u_clip(x, cy, 0.0, CLIP_HEIGHT))
        parts.append(make_bottom_disc(x, cy))

        # Nervure arrière pleine : liaison clip/plaque, évite le "dans le vide"
        spine = trimesh.creation.box(extents=[6.5, BACKPLATE_T + 1.2, CLIP_HEIGHT])
        spine.apply_translation([x, (BACKPLATE_T + 1.2) / 2.0, CLIP_HEIGHT / 2.0])
        parts.append(spine)

    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()

    # Coordonnées positives
    mesh.apply_translation(-mesh.bounds[0])
    return mesh


if __name__ == "__main__":
    mesh = build_model()
    mesh.export(OUTPUT)
    print("STL exporte :", OUTPUT)
    print("Dimensions finales mm :", mesh.extents)
