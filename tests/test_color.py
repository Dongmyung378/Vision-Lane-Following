"""Independent red/blue witnesses survive a recomputed swapped-RGB hash."""

import copy
import hashlib
import unittest

from carla_tools.color import ColorError, color_provenance, validate_color_provenance, verify_rgb_color
from scripts.carla_rgb_probe import rgb_from_bgra


class ColorTests(unittest.TestCase):
    def setUp(self):
        self.raw = bytes([1, 17, 231, 255, 241, 18, 3, 0, 31, 32, 33, 88, 4, 5, 6, 255])
        self.rgb, _ = rgb_from_bgra(self.raw, 2, 2)
        self.proof = color_provenance(self.raw, self.rgb, 2, 2)

    def test_bgra_to_rgb_proof_preserves_channel_order_and_ignores_alpha(self):
        self.assertEqual(self.rgb[:6], bytes([231, 17, 1, 3, 18, 241]))
        result = verify_rgb_color(self.rgb, self.proof, 2, 2, require_witness=True)
        self.assertTrue(result['red_blue_witness'])
        self.assertEqual(self.proof['source_sha256'], hashlib.sha256(self.raw).hexdigest())

    def test_red_blue_swap_is_refused_even_after_rgb_hash_recomputed(self):
        swapped = bytearray(self.rgb)
        swapped[0::3], swapped[2::3] = self.rgb[2::3], self.rgb[0::3]
        proof = copy.deepcopy(self.proof)
        proof['rgb_sha256'] = hashlib.sha256(swapped).hexdigest()
        with self.assertRaisesRegex(ColorError, 'bgra-rgb-channel-order-mismatch'):
            verify_rgb_color(bytes(swapped), proof, 2, 2)

    def test_damage_and_invalid_source_evidence_are_refused(self):
        with self.assertRaisesRegex(ColorError, 'decoded-rgb-hash-mismatch'):
            verify_rgb_color(bytes(len(self.rgb)), self.proof, 2, 2)
        mutations = [lambda p: p.update(source_format='RGBA8'),
                     lambda p: p['sample_pixels'][0].update(index=True),
                     lambda p: p['sample_pixels'][0]['bgra'].__setitem__(0, 256),
                     lambda p: p['sample_pixels'].append(copy.deepcopy(p['sample_pixels'][0])),
                     lambda p: p.update(extra=1), lambda p: p.update(rgb_sha256='secret')]
        for mutation in mutations:
            proof = copy.deepcopy(self.proof)
            mutation(proof)
            with self.subTest(mutation=mutation), self.assertRaises(ColorError):
                validate_color_provenance(proof, 2, 2)

    def test_greyscale_proof_cannot_certify_red_blue_order(self):
        raw = bytes([10, 10, 10, 255, 30, 30, 30, 255])
        rgb, _ = rgb_from_bgra(raw, 2, 1)
        proof = color_provenance(raw, rgb, 2, 1)
        self.assertFalse(verify_rgb_color(rgb, proof, 2, 1)['red_blue_witness'])
        with self.assertRaisesRegex(ColorError, 'no-red-blue-distinguishing-pixel'):
            verify_rgb_color(rgb, proof, 2, 1, require_witness=True)


if __name__ == '__main__':
    unittest.main()
