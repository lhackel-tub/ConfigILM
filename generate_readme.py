import json
from requests import get

COLOR_THRESHOLDS = [
    (101, 95, '4c1'),  # 101 is higher than the max coverage, needed for easier comparison
    (95, 90, '97CA00'),
    (90, 75, 'a4a61d'),
    (75, 60, 'dfb317'),
    (60, 40, 'fe7d37'),
    (40, 0, 'e05d44'),
]

if __name__ == "__main__":
    # Read the template file
    with open("README_template.md", "r") as file:
        template = file.read()

    # Read the coverage json file
    with open("coverage.json", "r") as file:
        coverage = json.load(file)
    overall_coverage = int(coverage["totals"]["percent_covered_display"])
    print(f"Overall coverage: {overall_coverage}")
    # get the correct color for the coverage
    color = [color for upper, lower, color in COLOR_THRESHOLDS if lower <= overall_coverage < upper][0]
    print(f"Color: {color}")
    coverage_str = f'https://img.shields.io/badge/coverage%20-{overall_coverage}%25-{color}'
    # Replace the placeholders in the template
    readme = template.replace("<COVERAGE_BADGE_LINK>", coverage_str)

    r = get("https://zenodo.org/doi/10.5281/zenodo.7767950")
    raw = r.text

    # find zenodo link that contains the substring '<meta property="og:url" content="{link}" />'
    zenodo_link = raw.split('<meta property="og:url" content="')[1].split('" />')[0]
    print(zenodo_link)
    readme = readme.replace("<CURRENT_ZENODO_LINK>", zenodo_link)

    # find zenodo id
    zenodo_id = zenodo_link.split("/")[-1]
    print(zenodo_id)
    badge_link = f'https://zenodo.org/badge/DOI/10.5281/zenodo.{zenodo_id}.svg'
    readme = readme.replace("<CURRENT_ZENODO_BADGE>", badge_link)

    # Get the bibtex info from zenodo
    r = get(f"https://zenodo.org/records/{zenodo_id}/export/bibtex")
    bibtex = r.text
    print(bibtex)
    readme = readme.replace("<CURRENT_ZENODO_BIBTEX_INFO>", bibtex)

    # Write the new README
    with open("README.md", "w") as file:
        file.write(readme)