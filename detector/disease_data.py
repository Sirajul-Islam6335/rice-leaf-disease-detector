"""Seed data for the `Disease` SQL table (loaded by migration 0002 and `manage.py seed_diseases`).

Keys must match the class names stored in the model checkpoint (the dataset's
folder names). The guidance is general background for screening; local
agricultural extension services should confirm a diagnosis and recommend
products registered in the grower's country.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Disease:
    name: str
    slug: str
    pathogen: str
    kind: str
    color: str            # symptom-derived colour used for this class everywhere in the UI
    sample: str           # static path of a sample photo (from the held-out test split)
    summary: str
    symptoms: list = field(default_factory=list)
    favoured_by: list = field(default_factory=list)
    management: list = field(default_factory=list)
    looks_like: str = ""


DISEASES = {
    "Bacterial leaf blight": Disease(
        name="Bacterial leaf blight",
        slug="bacterial-leaf-blight",
        pathogen="Xanthomonas oryzae pv. oryzae",
        kind="Bacterial",
        color="#B8860B",
        sample="detector/img/samples/bacterial-leaf-blight.jpg",
        summary=(
            "One of the most damaging rice diseases in irrigated and rain-fed lowlands. "
            "The bacterium enters through wounds and water pores and spreads along the leaf's veins."
        ),
        symptoms=[
            "Water-soaked streaks that start at the leaf tip or edges and spread down the blade",
            "Lesions turn yellow, then straw-coloured or greyish-white, often with a wavy margin",
            "Whole leaves can dry out; young plants may wilt completely (known as kresek)",
            "Milky or yellowish bacterial droplets on young lesions early in the morning",
        ],
        favoured_by=[
            "Warm, humid weather with rain and wind, especially after storms",
            "Heavy nitrogen fertiliser",
            "Wounded leaves, deep flooding and irrigation water carrying bacteria between fields",
        ],
        management=[
            "Plant resistant varieties where they are available; this is the most effective control",
            "Use balanced fertiliser and split nitrogen applications instead of one heavy dose",
            "Keep fields and bunds clean: remove infected stubble, straw, ratoons and grassy weeds",
            "Drain fields periodically and avoid moving water from infected fields into healthy ones",
            "Chemical sprays are of limited use against this bacterium; ask your extension service before spraying",
        ],
        looks_like="Can be confused with nutrient scorch or drought-dried leaf tips, which lack the wavy, water-soaked margin.",
    ),
    "Brown spot": Disease(
        name="Brown spot",
        slug="brown-spot",
        pathogen="Bipolaris oryzae (Cochliobolus miyabeanus)",
        kind="Fungal",
        color="#8A4B22",
        sample="detector/img/samples/brown-spot.jpg",
        summary=(
            "A fungal disease closely tied to stressed, poorly nourished crops. "
            "It reduces grain filling and quality, and can be severe on nutrient-poor or drought-prone soils."
        ),
        symptoms=[
            "Small, round to oval brown spots scattered over the leaf blade",
            "Mature spots have a grey or pale centre and a dark reddish-brown edge, often with a yellow halo",
            "Spots can merge and dry out large parts of the leaf",
            "Dark spots may also appear on the husks of the grain",
        ],
        favoured_by=[
            "Soils low in nutrients, especially potassium, silicon or nitrogen",
            "Drought stress and irregular water supply",
            "Infected seed, and long periods of leaf wetness",
        ],
        management=[
            "Correct the soil's nutrient balance, particularly potassium and silicon",
            "Sow clean, certified seed; seed treatment reduces early infection",
            "Keep water supply steady to avoid drought stress",
            "Use tolerant varieties and remove infected crop residue",
            "Fungicides can help in severe outbreaks; follow local recommendations",
        ],
        looks_like="Most often confused with leaf smut and with blast; blast lesions are diamond-shaped with pointed ends.",
    ),
    "Leaf smut": Disease(
        name="Leaf smut",
        slug="leaf-smut",
        pathogen="Entyloma oryzae",
        kind="Fungal",
        color="#3B3F45",
        sample="detector/img/samples/leaf-smut.jpg",
        summary=(
            "A widespread but usually minor fungal disease that appears late in the season. "
            "It rarely causes serious yield loss on its own, but heavy infection can make leaves dry out early."
        ),
        symptoms=[
            "Small, slightly raised black spots, often angular or slightly elongated",
            "Spots appear on both sides of the leaf, frequently lined up between the veins",
            "Heavily infected leaf tips may turn yellow, dry out and split",
        ],
        favoured_by=[
            "High nitrogen fertiliser",
            "Infected leaf debris left in the field, which carries the fungus to the next crop",
        ],
        management=[
            "Usually not worth spraying on its own; focus on good crop management",
            "Avoid excessive nitrogen",
            "Remove or plough in infected residue after harvest",
            "Use resistant varieties where leaf smut is common",
        ],
        looks_like="Easily confused with early brown spot; leaf smut spots are smaller, blacker and raised, without a pale centre.",
    ),
}






