"""Prepare new XSum articles for source-grounded target curation; no GPU needed."""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from summarization.config import load_config

# Previously seen training/review/evaluation IDs from the supplied project exports.
KNOWN_IDS = frozenset(('10693692', '13345166', '13629860', '14084455', '14094760', '14292072', '14595368', '14637701', '14951350', '15554361', '15682606', '16441633', '16849434', '17181327', '17532613', '17989371', '18338838', '18490935', '18954311', '19046923', '19087826', '19475087', '20126971', '20211138', '20621350', '20896049', '21123204', '21187585', '21341485', '21467982', '21938080', '21958369', '22327445', '22660723', '22776802', '22839609', '23205638', '23221293', '23495026', '23593126', '23602056', '23623157', '23775475', '23839017', '23899019', '24436781', '24573605', '24842750', '24923879', '25166283', '25765975', '25779937', '25903685', '25903959', '26286799', '26316092', '26557057', '26967183', '27156065', '27163373', '27502137', '27533169', '27533389', '27588847', '27720617', '28019350', '28038432', '28511964', '28755287', '29003321', '29011794', '29054062', '29061972', '29064989', '29300134', '29321143', '29361043', '29507419', '29685642', '29743080', '29843081', '29867225', '29921363', '30192013', '30292532', '30312462', '30367734', '30379295', '30431988', '30437661', '30494612', '30532822', '30573259', '30636021', '30655028', '30731863', '30778452', '30878568', '30908073', '30922486', '31047271', '31051223', '31072799', '31373711', '31498508', '31530667', '31534284', '31583949', '31603031', '31655696', '31682417', '31830941', '31835356', '31861022', '31908080', '31989242', '32036322', '32069441', '32083717', '32122134', '32147474', '32231079', '32233570', '32250342', '32284473', '32291349', '32298797', '32338905', '32346790', '32368134', '32419285', '32430816', '32489357', '32499267', '32551680', '32582528', '32610327', '32621768', '32625796', '32655725', '32695546', '32782647', '32784395', '32802276', '32880367', '32884747', '32934264', '32964870', '32986900', '32993571', '32994511', '33014991', '33044213', '33080177', '33096106', '33122308', '33132613', '33188535', '33227143', '33363265', '33400425', '33434222', '33604811', '33624588', '33686121', '33717908', '33733320', '33782863', '33794961', '33814748', '33819596', '33863435', '33886180', '33928485', '33998661', '34013736', '34044707', '34052190', '34065020', '34084194', '34140195', '34169957', '34173752', '34184529', '34186518', '34223705', '34363448', '34401815', '34423411', '34424982', '34499525', '34503579', '34504013', '34519767', '34539507', '34560764', '34566103', '34583459', '34606633', '34609616', '34613957', '34675020', '34676841', '34677344', '34696346', '34699304', '34720961', '34743185', '34806376', '34854895', '34859242', '34864393', '34897767', '34924097', '34971925', '34973771', '34976582', '35006667', '35036869', '35058974', '35082705', '35087050', '35111542', '35119567', '35158817', '35159256', '35159826', '35167744', '35231360', '35237762', '35259359', '35260931', '35305028', '35331251', '35350418', '35389201', '35421179', '35427552', '35483754', '35490204', '35515193', '35517984', '35528653', '35550637', '35566956', '35566972', '35568054', '35571054', '35587330', '35587664', '35606413', '35633915', '35649790', '35651943', '35689952', '35693853', '35714493', '35736684', '35744418', '35773460', '35782332', '35799835', '35808776', '35814593', '35836919', '35849640', '35849641', '35850869', '35858564', '35859402', '35903736', '35906890', '35911572', '35931696', '35948868', '35952861', '35997110', '36015489', '36023329', '36038661', '36047925', '36095804', '36112388', '36135901', '36144172', '36152447', '36171867', '36181218', '36199993', '36212050', '36220327', '36230808', '36264092', '36272269', '36275487', '36287551', '36297780', '36339063', '36342383', '36376491', '36398922', '36407030', '36459014', '36469365', '36471493', '36479190', '36481452', '36488292', '36500312', '36558359', '36558761', '36685359', '36688213', '36718950', '36727684', '36739372', '36745362', '36745656', '36757446', '36761220', '36781876', '36782342', '36819350', '36820021', '36822927', '36835454', '36851587', '36852365', '36889950', '36899343', '36900286', '36914334', '36921682', '36930699', '36930808', '36950083', '36974551', '36998861', '37058129', '37081934', '37109867', '37151620', '37158402', '37165190', '37172535', '37228870', '37263954', '37276883', '37290297', '37300272', '37387356', '37392066', '37452204', '37453824', '37454791', '37482950', '37485589', '37506251', '37542868', '37560834', '37569795', '37574881', '37580644', '37585188', '37594267', '37600158', '37601422', '37603857', '37628585', '37643894', '37653065', '37710786', '37811472', '37829830', '37844632', '37895601', '37906596', '37978482', '37987696', '37988095', '37990937', '38011196', '38032989', '38047171', '38062376', '38074177', '38084449', '38091766', '38102507', '38124653', '38139925', '38142816', '38144547', '38147005', '38167560', '38195866', '38231096', '38284727', '38313110', '38325800', '38488369', '38529237', '38530543', '38536025', '38561031', '38598310', '38638845', '38639241', '38654026', '38655061', '38662225', '38674176', '38689143', '38740717', '38796393', '38804692', '38809875', '38898662', '39016181', '39020706', '39030031', '39067028', '39113833', '39141659', '39145361', '39169601', '39197677', '39203549', '39255971', '39279677', '39318455', '39329304', '39337805', '39351813', '39355604', '39365690', '39374630', '39387550', '39429069', '39441826', '39472450', '39501904', '39502096', '39514604', '39546632', '39551463', '39555727', '39559471', '39577667', '39600613', '39635070', '39642497', '39681808', '39687563', '39691925', '39706990', '39733851', '39755721', '39770395', '39778169', '39788959', '39790507', '39799514', '39807676', '39820434', '39836394', '39843775', '39865291', '39869299', '39917330', '39978284', '40030337', '40160081', '40194767', '40233713', '40242974', '40251760', '40277696', '40285980', '40297394', '40300734', '40340492', '40352334', '40362716', '40387347', '40418367', '40442280', '40464768', '40523487', '40542374', '40582486', '40601366', '40606261', '40606364', '40627394', '40634756', '40641376', '40653524', '40685087', '40691894', '40750170', '40753182', '40774742', '40828370', '40854127', '40857379', '40863406', '40916652', '40924207', '40926837', '40959959', '40972546', '41006539', '41118514', "Value(dtype='string', id=None)"))

REVIEW_INSTRUCTIONS = """Review every article independently using only its article text.
Return one downloadable JSON file, keeping metadata and all IDs/articles unchanged.
For each example:
- Write reviewed_summary as a concise English summary, usually 25-40 words and
  one or two complete sentences. Shorter summaries are appropriate for short news.
- Select the main event or finding. Omit peripheral details. Preserve who did
  what, dates, quantities, attribution, uncertainty, negation and causal relations.
- Do not use outside knowledge or the original BBC lead. Do not infer a person's
  first name, age, charge, motive, organisation or location if unsupported.
- Natural reuse of names and technical terms is allowed. Do not mechanically
  copy the opening or force facts from every part of the article into the summary.
- Provide source_evidence as a list of objects with claim and quote. Each quote
  must be an exact non-empty substring of article, and support its associated
  claim in context. Cover every factual claim in reviewed_summary.
- If the article cannot support a useful summary, set review_status=withheld,
  reviewed_summary=null and explain in review_note. Otherwise use review_status=draft.
- Keep training_eligible=false and human_reviewed=false in all rows. These are
  review candidates; downstream adjudication must approve training eligibility.
Return every example exactly once, without placeholders or omitted articles.
"""


def collect_ids(value):
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"id", "doc_id"} and isinstance(item, (str, int)):
                result.add(str(item))
            else:
                result.update(collect_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(collect_ids(item))
    return result


def valid_text(value):
    return isinstance(value, str) and bool(value.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        parser.error("count and batch-size must be positive")

    config = load_config()
    paths, settings = config["paths"], config["data"]
    destination = args.output or paths["data_dir"] / "curation" / "expansion_1000_v1"
    if destination.exists():
        raise FileExistsError(f"Output already exists; keeping it intact: {destination}")
    dataset_path = paths["data_dir"] / "processed" / "xsum"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {dataset_path}")

    excluded = set(KNOWN_IDS)
    for directory in (paths["reports_dir"], paths["data_dir"] / "curation"):
        if directory.exists():
            for file in directory.rglob("*.json"):
                try:
                    excluded.update(collect_ids(json.loads(file.read_text(encoding="utf-8"))))
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f"Cannot read exclusion file: {file}") from error

    from datasets import load_from_disk
    dataset = load_from_disk(str(dataset_path))
    train = dataset["train"]
    source_col, id_col = settings["source_column"], settings["id_column"]
    for name in (source_col, id_col):
        if name not in train.column_names:
            raise ValueError(f"Missing training column: {name}")
    heldout_docs, heldout_ids = set(), set()
    for split in ("validation", "test"):
        heldout_docs.update(d for d in dataset[split][source_col] if valid_text(d))
        heldout_ids.update(str(i) for i in dataset[split][id_col])
    excluded.update(heldout_ids)

    order = list(range(len(train)))
    seed = config["project"]["seed"]
    random.Random(seed).shuffle(order)
    examples, seen_docs, selected_ids = [], set(), set()
    for index in order:
        row = train[index]
        article, identifier = row[source_col], str(row[id_col])
        if not valid_text(article) or identifier in excluded or identifier in selected_ids:
            continue
        if article in heldout_docs or article in seen_docs:
            continue
        examples.append({
            "id": identifier, "article": article, "reviewed_summary": None,
            "source_evidence": [], "review_status": "pending",
            "training_eligible": False, "human_reviewed": False,
            "review_note": None,
        })
        seen_docs.add(article)
        selected_ids.add(identifier)
        if len(examples) == args.count:
            break
    if len(examples) != args.count:
        raise ValueError(f"Only {len(examples)} eligible new articles found")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": settings["dataset_name"],
        "configured_dataset_revision": settings["revision"],
        "cleaned_train_fingerprint": getattr(train, "_fingerprint", None),
        "split": "train", "seed": seed, "examples": len(examples),
        "excluded_ids": len(excluded), "original_summaries_supplied": False,
        "selection": "Seeded random selection from cleaned train; exact document overlap excluded.",
        "near_duplicates_checked": False,
    }
    destination.mkdir(parents=True)
    batches = []
    for start in range(0, len(examples), args.batch_size):
        batch_number = start // args.batch_size + 1
        name = f"curation_batch_{batch_number:02d}.json"
        payload = {
            "metadata": {**metadata, "batch": batch_number},
            "instructions": REVIEW_INSTRUCTIONS,
            "examples": examples[start:start + args.batch_size],
        }
        (destination / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        batches.append(name)
    manifest = {**metadata, "selected_ids": sorted(selected_ids), "batches": batches}
    manifest["selection_sha256"] = hashlib.sha256(
        json.dumps(examples, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    (destination / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(examples)} NEW articles in {len(batches)} review batches.")
    print(f"Saved: {destination}")
    print("Targets remain pending review. No training or model download was started.")


if __name__ == "__main__":
    main()
