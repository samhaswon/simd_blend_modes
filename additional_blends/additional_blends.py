import numpy as np


def _ensure_rgba(image):
    if image.shape[2] == 4:
        return image

    alpha = np.full(image.shape[:2] + (1,), 255.0, dtype=image.dtype)
    return np.concatenate((image, alpha), axis=2)


def _compose_alpha(img_in, img_layer, opacity):
    comp_alpha = np.minimum(img_in[..., 3], img_layer[..., 3]) * opacity
    new_alpha = img_in[..., 3] + (1.0 - img_in[..., 3]) * comp_alpha

    with np.errstate(divide="ignore", invalid="ignore"):
        comp_alpha /= new_alpha

    comp_alpha[np.isnan(comp_alpha)] = 0.0
    return comp_alpha


def _rgb_to_hsv(rgb):
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]

    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    chroma = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    red_mask = (max_channel == red) & (chroma != 0.0)
    green_mask = (max_channel == green) & (chroma != 0.0)
    blue_mask = (max_channel == blue) & (chroma != 0.0)

    hue[red_mask] = ((green[red_mask] - blue[red_mask]) / chroma[red_mask]) % 6.0
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / chroma[green_mask]) + 2.0
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / chroma[blue_mask]) + 4.0
    hue /= 6.0

    saturation = np.zeros_like(max_channel)
    nonzero_value = max_channel != 0.0
    saturation[nonzero_value] = chroma[nonzero_value] / max_channel[nonzero_value]

    return np.dstack((hue, saturation, max_channel))


def _hsv_to_rgb(hsv):
    hue = (hsv[..., 0] % 1.0) * 6.0
    saturation = np.clip(hsv[..., 1], 0.0, 1.0)
    value = np.clip(hsv[..., 2], 0.0, 1.0)

    chroma = value * saturation
    x_value = chroma * (1.0 - np.abs((hue % 2.0) - 1.0))
    match = value - chroma

    rgb_prime = np.zeros_like(hsv)
    masks = [
        (0.0 <= hue) & (hue < 1.0),
        (1.0 <= hue) & (hue < 2.0),
        (2.0 <= hue) & (hue < 3.0),
        (3.0 <= hue) & (hue < 4.0),
        (4.0 <= hue) & (hue < 5.0),
        (5.0 <= hue) & (hue < 6.0),
    ]
    values = [
        (chroma, x_value, 0.0),
        (x_value, chroma, 0.0),
        (0.0, chroma, x_value),
        (0.0, x_value, chroma),
        (x_value, 0.0, chroma),
        (chroma, 0.0, x_value),
    ]

    for mask, channels in zip(masks, values):
        for channel_index, channel_value in enumerate(channels):
            if np.isscalar(channel_value):
                rgb_prime[..., channel_index][mask] = channel_value
            else:
                rgb_prime[..., channel_index][mask] = channel_value[mask]

    return rgb_prime + np.expand_dims(match, axis=2)


def _rgb_to_hsl(rgb):
    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    lightness = (max_channel + min_channel) / 2.0
    chroma = max_channel - min_channel

    hsv = _rgb_to_hsv(rgb)
    saturation = np.zeros_like(lightness)
    saturation_mask = chroma != 0.0
    saturation[saturation_mask] = (
        chroma[saturation_mask]
        / (1.0 - np.abs(2.0 * lightness[saturation_mask] - 1.0))
    )

    return np.dstack((hsv[..., 0], saturation, lightness))


def _hsl_to_rgb(hsl):
    hue = hsl[..., 0] % 1.0
    saturation = np.clip(hsl[..., 1], 0.0, 1.0)
    lightness = np.clip(hsl[..., 2], 0.0, 1.0)

    chroma = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
    value = lightness + chroma / 2.0
    hsv_saturation = np.zeros_like(saturation)
    value_mask = value != 0.0
    hsv_saturation[value_mask] = chroma[value_mask] / value[value_mask]

    return _hsv_to_rgb(np.dstack((hue, hsv_saturation, value)))


def _srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * (rgb ** (1.0 / 2.4)) - 0.055,
    )


def _xyz_to_lab_component(value):
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    return np.where(value > epsilon, np.cbrt(value), (kappa * value + 16.0) / 116.0)


def _lab_to_xyz_component(value):
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    value_cubed = value ** 3.0
    return np.where(value_cubed > epsilon, value_cubed, (116.0 * value - 16.0) / kappa)


def _rgb_to_lab(rgb):
    linear = _srgb_to_linear(rgb)
    x = (
        linear[..., 0] * 0.4124564
        + linear[..., 1] * 0.3575761
        + linear[..., 2] * 0.1804375
    )
    y = (
        linear[..., 0] * 0.2126729
        + linear[..., 1] * 0.7151522
        + linear[..., 2] * 0.0721750
    )
    z = (
        linear[..., 0] * 0.0193339
        + linear[..., 1] * 0.1191920
        + linear[..., 2] * 0.9503041
    )

    fx = _xyz_to_lab_component(x / 0.95047)
    fy = _xyz_to_lab_component(y)
    fz = _xyz_to_lab_component(z / 1.08883)

    lightness = 116.0 * fy - 16.0
    a_channel = 500.0 * (fx - fy)
    b_channel = 200.0 * (fy - fz)
    return np.dstack((lightness, a_channel, b_channel))


def _lab_to_rgb(lab):
    lightness = lab[..., 0]
    a_channel = lab[..., 1]
    b_channel = lab[..., 2]

    fy = (lightness + 16.0) / 116.0
    fx = fy + a_channel / 500.0
    fz = fy - b_channel / 200.0

    x = 0.95047 * _lab_to_xyz_component(fx)
    y = _lab_to_xyz_component(fy)
    z = 1.08883 * _lab_to_xyz_component(fz)

    red = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    green = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    blue = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    return _linear_to_srgb(np.dstack((red, green, blue)))


def _lab_to_lch(lab):
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.arctan2(lab[..., 2], lab[..., 1])
    return np.dstack((lab[..., 0], chroma, hue))


def _lch_to_lab(lch):
    a_channel = lch[..., 1] * np.cos(lch[..., 2])
    b_channel = lch[..., 1] * np.sin(lch[..., 2])
    return np.dstack((lch[..., 0], a_channel, b_channel))


def _blend_rgb(img_in, img_layer, opacity, blend_func):
    img_in = _ensure_rgba(img_in)
    img_layer = _ensure_rgba(img_layer)

    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0

    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)

    comp = blend_func(img_in_norm[..., :3], img_layer_norm[..., :3])
    ratio_rs = np.expand_dims(ratio, axis=2)
    img_out = comp * ratio_rs + img_in_norm[..., :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[..., 3])))

    return img_out * 255.0


def _hsv_blend(img_in, img_layer, channels):
    in_hsv = _rgb_to_hsv(img_in)
    layer_hsv = _rgb_to_hsv(img_layer)
    out_hsv = in_hsv.copy()
    if channels == 0:
        saturation_mask = layer_hsv[..., 1] != 0.0
        out_hsv[..., 0][saturation_mask] = layer_hsv[..., 0][saturation_mask]
    else:
        out_hsv[..., channels] = layer_hsv[..., channels]
    return _hsv_to_rgb(out_hsv)


def _lch_blend(img_in, img_layer, channels):
    in_lch = _lab_to_lch(_rgb_to_lab(img_in))
    layer_lch = _lab_to_lch(_rgb_to_lab(img_layer))
    out_lch = in_lch.copy()
    if channels == 2:
        chroma_mask = layer_lch[..., 1] > 0.0
        out_lch[..., 2][chroma_mask] = layer_lch[..., 2][chroma_mask]
    else:
        out_lch[..., channels] = layer_lch[..., channels]
    return _lab_to_rgb(_lch_to_lab(out_lch))


def hsv_hue(img_in, img_layer, opacity):
    """Apply HSV hue blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: _hsv_blend(bg, fg, 0))


def hsv_saturation(img_in, img_layer, opacity):
    """Apply HSV saturation blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: _hsv_blend(bg, fg, 1))


def hsv_value(img_in, img_layer, opacity):
    """Apply HSV value blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: _hsv_blend(bg, fg, 2))


def hsl_color(img_in, img_layer, opacity):
    """Apply HSL color blend mode."""

    def blend(bg, fg):
        bg_hsl = _rgb_to_hsl(bg)
        fg_hsl = _rgb_to_hsl(fg)
        return _hsl_to_rgb(
            np.dstack((fg_hsl[..., 0], fg_hsl[..., 1], bg_hsl[..., 2]))
        )

    return _blend_rgb(img_in, img_layer, opacity, blend)


def lch_hue(img_in, img_layer, opacity):
    """Apply LCh hue blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: _lch_blend(bg, fg, 2))


def lch_chroma(img_in, img_layer, opacity):
    """Apply LCh chroma blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: _lch_blend(bg, fg, 1))


def lch_color(img_in, img_layer, opacity):
    """Apply LCh color blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: _lch_blend(bg, fg, [1, 2]))


def lch_lightness(img_in, img_layer, opacity):
    """Apply LCh lightness blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: _lch_blend(bg, fg, 0))


def burn(img_in, img_layer, opacity):
    """Apply color burn blend mode."""
    def blend(bg, fg):
        with np.errstate(divide="ignore", invalid="ignore"):
            burned = 1.0 - np.minimum(1.0, (1.0 - bg) / fg)
        return np.where(fg == 0.0, np.where(bg == 1.0, 1.0, 0.0), burned)

    return _blend_rgb(img_in, img_layer, opacity, blend)


def linear_burn(img_in, img_layer, opacity):
    """Apply linear burn blend mode."""
    return _blend_rgb(
        img_in,
        img_layer,
        opacity,
        lambda bg, fg: np.maximum(bg + fg - 1.0, 0.0),
    )


def exclusion(img_in, img_layer, opacity):
    """Apply exclusion blend mode."""
    return _blend_rgb(img_in, img_layer, opacity, lambda bg, fg: bg + fg - 2.0 * bg * fg)


def vivid_light(img_in, img_layer, opacity):
    """Apply vivid light blend mode."""

    def blend(bg, fg):
        burn_layer = 2.0 * fg
        dodge_layer = 2.0 * (fg - 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            burned = np.where(
                burn_layer == 0.0,
                np.where(bg == 1.0, 1.0, 0.0),
                1.0 - np.minimum(1.0, (1.0 - bg) / burn_layer),
            )
            dodged = np.where(
                dodge_layer == 1.0,
                1.0,
                np.minimum(1.0, bg / (1.0 - dodge_layer)),
            )
        return np.where(fg < 0.5, burned, dodged)

    return _blend_rgb(img_in, img_layer, opacity, blend)


def pin_light(img_in, img_layer, opacity):
    """Apply pin light blend mode."""

    def blend(bg, fg):
        return np.where(
            fg < 0.5,
            np.minimum(bg, 2.0 * fg),
            np.maximum(bg, 2.0 * fg - 1.0),
        )

    return _blend_rgb(img_in, img_layer, opacity, blend)
