"""The original fixed three-class prediction target."""

VIEW_CUTOFFS = (100_000, 1_000_000)
VIEW_NAMES = ("หลักหมื่น", "หลักแสน", "หลักล้าน")


def view_band(views: int) -> int:
    return sum(views >= cutoff for cutoff in VIEW_CUTOFFS)
