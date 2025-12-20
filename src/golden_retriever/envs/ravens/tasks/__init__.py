"""Ravens tasks."""

from .align_box_corner import AlignBoxCorner
from .align_rope import AlignRope
from .assembling_kits import AssemblingKits, AssemblingKitsEasy
from .assembling_kits_seq import (
    AssemblingKitsSeqFull,
    AssemblingKitsSeqSeenColors,
    AssemblingKitsSeqUnseenColors,
)
from .block_insertion import (
    BlockInsertion,
    BlockInsertionEasy,
    BlockInsertionNoFixture,
    BlockInsertionSixDof,
    BlockInsertionTranslation,
)
from .manipulating_rope import ManipulatingRope
from .new_put_blocks_different_corners import PutBlocksDifferentCorners
from .new_put_blocks_matching_colors import PutBlocksMatchingColors
from .new_put_blocks_mismatched_colors import PutBlocksMismatchedColors
from .new_put_blocks_on_corner_side import PutBlocksOnCornerSide
from .new_put_letters_alphabetical_order import PutLettersAlphabeticalOrder
from .new_put_letters_reverse_alphabetical_order import (
    PutLettersReverseAlphabeticalOrder,
)
from .new_separate_consonants import SeparateConsonants
from .new_separate_vowels import SeparateVowels
from .new_sort_letters_less_than_d import SortLettersLessThanD
from .new_sort_primary_color_blocks import SortPrimaryColorBlocks
from .new_sort_symmetrical_letters import SortSymmetricalLetters
from .new_spell_sport import SpellSport
from .new_spell_word import SpellWord
from .new_stack_blocks import StackBlocks
from .new_stack_blocks_cool_colors import StackBlocksCoolColors
from .new_stack_blocks_warm_colors import StackBlocksWarmColors
from .packing_boxes import PackingBoxes
from .packing_boxes_pairs import (
    PackingBoxesPairsFull,
    PackingBoxesPairsSeenColors,
    PackingBoxesPairsUnseenColors,
)
from .packing_google_objects import (
    PackingSeenGoogleObjectsGroup,
    PackingSeenGoogleObjectsSeq,
    PackingUnseenGoogleObjectsGroup,
    PackingUnseenGoogleObjectsSeq,
)
from .packing_shapes import PackingShapes
from .palletizing_boxes import PalletizingBoxes
from .place_red_in_green import PlaceRedInGreen
from .put_block_in_bowl import (
    PutBlockInBowlFull,
    PutBlockInBowlSeenColors,
    PutBlockInBowlUnseenColors,
)
from .separating_piles import (
    SeparatingPilesFull,
    SeparatingPilesSeenColors,
    SeparatingPilesUnseenColors,
)
from .stack_block_pyramid import StackBlockPyramid
from .stack_block_pyramid_seq import (
    StackBlockPyramidSeqFull,
    StackBlockPyramidSeqSeenColors,
    StackBlockPyramidSeqUnseenColors,
)
from .sweeping_piles import SweepingPiles
from .task import Task
from .towers_of_hanoi import TowersOfHanoi
from .towers_of_hanoi_seq import (
    TowersOfHanoiSeqFull,
    TowersOfHanoiSeqSeenColors,
    TowersOfHanoiSeqUnseenColors,
)

names = {
    # demo conditioned (original Transporter Networks paper)
    "align-box-corner": AlignBoxCorner,
    "assembling-kits": AssemblingKits,
    "assembling-kits-easy": AssemblingKitsEasy,
    "block-insertion": BlockInsertion,
    "block-insertion-easy": BlockInsertionEasy,
    "block-insertion-nofixture": BlockInsertionNoFixture,
    "block-insertion-sixdof": BlockInsertionSixDof,
    "block-insertion-translation": BlockInsertionTranslation,
    "manipulating-rope": ManipulatingRope,
    "packing-boxes": PackingBoxes,
    "palletizing-boxes": PalletizingBoxes,
    "place-red-in-green": PlaceRedInGreen,
    "stack-block-pyramid": StackBlockPyramid,
    "sweeping-piles": SweepingPiles,
    "towers-of-hanoi": TowersOfHanoi,
    # goal conditioned (CLIPort paper)
    "align-rope": AlignRope,
    "assembling-kits-seq-seen-colors": AssemblingKitsSeqSeenColors,
    "assembling-kits-seq-unseen-colors": AssemblingKitsSeqUnseenColors,
    "assembling-kits-seq-full": AssemblingKitsSeqFull,
    "packing-shapes": PackingShapes,
    "packing-boxes-pairs-seen-colors": PackingBoxesPairsSeenColors,
    "packing-boxes-pairs-unseen-colors": PackingBoxesPairsUnseenColors,
    "packing-boxes-pairs-full": PackingBoxesPairsFull,
    "packing-seen-google-objects-seq": PackingSeenGoogleObjectsSeq,
    "packing-unseen-google-objects-seq": PackingUnseenGoogleObjectsSeq,
    "packing-seen-google-objects-group": PackingSeenGoogleObjectsGroup,
    "packing-unseen-google-objects-group": PackingUnseenGoogleObjectsGroup,
    "put-block-in-bowl-seen-colors": PutBlockInBowlSeenColors,
    "put-block-in-bowl-unseen-colors": PutBlockInBowlUnseenColors,
    "put-block-in-bowl-full": PutBlockInBowlFull,
    "stack-block-pyramid-seq-seen-colors": StackBlockPyramidSeqSeenColors,
    "stack-block-pyramid-seq-unseen-colors": StackBlockPyramidSeqUnseenColors,
    "stack-block-pyramid-seq-full": StackBlockPyramidSeqFull,
    "separating-piles-seen-colors": SeparatingPilesSeenColors,
    "separating-piles-unseen-colors": SeparatingPilesUnseenColors,
    "separating-piles-full": SeparatingPilesFull,
    "towers-of-hanoi-seq-seen-colors": TowersOfHanoiSeqSeenColors,
    "towers-of-hanoi-seq-unseen-colors": TowersOfHanoiSeqUnseenColors,
    "towers-of-hanoi-seq-full": TowersOfHanoiSeqFull,
    # my new tasks (Blocks & Bowls)
    "stack-blocks": StackBlocks,
    "put-blocks-on-corner-side": PutBlocksOnCornerSide,
    "put-blocks-matching-colors": PutBlocksMatchingColors,
    "put-blocks-mismatched-colors": PutBlocksMismatchedColors,
    "put-blocks-different-corners": PutBlocksDifferentCorners,
    "stack-blocks-cool-colors": StackBlocksCoolColors,
    "stack-blocks-warm-colors": StackBlocksWarmColors,
    "sort-primary-color-blocks": SortPrimaryColorBlocks,
    # my new tasks (Letters)
    "put-letters-alphabetical-order": PutLettersAlphabeticalOrder,
    "spell-word": SpellWord,
    "separate-vowels": SeparateVowels,
    "put-letters-reverse-alphabetical-order": PutLettersReverseAlphabeticalOrder,
    "spell-sport": SpellSport,
    "sort-symmetrical-letters": SortSymmetricalLetters,
    "separate-consonants": SeparateConsonants,
    "sort-letters-less-than-d": SortLettersLessThanD,
}
