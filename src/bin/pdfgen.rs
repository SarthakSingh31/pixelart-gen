use std::{
    collections::{hash_map::RandomState, HashMap},
    fs,
    io::BufWriter,
    ops::Range,
    path::PathBuf,
};

use clap::Parser;
use glam::{DVec2, UVec2};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use palette::{chromatic_adaptation::AdaptFrom, color_difference::EuclideanDistance};
use printpdf::{
    ImageTransform, IndirectFontRef, Line, Mm, PdfDocument, PdfDocumentReference,
    PdfLayerReference, Point,
};

const SYMBOLS: [char; 200] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n',
    'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'y', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽', '❾', '❿', '➀', '➁', '➂', '➃', '➄', '➅', '➆', '➇', '➈',
    '➉', '~', '!', '@', '#', '$', '%', '&', '*', '+', '=', '✇', '✈', '✉', '✎', '✒', '✓', '✖', '✜',
    '✢', '✥', '✦', '✩', '✲', '✵', '✹', '✺', '✼', '✾', '✿', '❀', '❁', '❄', '❈', '❍', '❑', '❖', '❢',
    '❤', '❦', '➔', '➘', '➢', '➥', '➲', '➳', '➺', '➾', '◒', '◐', '◍', '◌', '◉', '◈', '▤', '▧', '◆',
    '◇', '◔', '◗', '◘', '⌘', '⍾', '⏏', '␥', '◩', '☂', '☘', '⟰', '⟲', '⟴', '⤀', '⤄', '⤒', '⤙', '⤝',
    '⤡', '⤧', '⤴', '⤹', '⥋', '⥐', '⥽', '⦁', '⦂', '⦊', '⦔', '⦛', '⦵', '⦶', '⩁', '⦸', '⦹', '⩐', '⦻',
    '⦼', '⦾', '⧀', '⧄', '⧆', '⩆', '⩌', '⩎', '⧍', '⧑', '⧖', '⧜', '⧝', '⧞', '⧢', '⧥', '⧨', '⧫', '⧬',
    '⧮', '⧲', '⨀', '⨁', '⨇', '⨊', '⨎', '⨳', '⨷', '⨿',
];

const REGULAR: &[u8] = include_bytes!("/usr/share/fonts/noto/NotoSans-Regular.ttf");
const BOLD: &[u8] = include_bytes!("/usr/share/fonts/noto/NotoSans-Bold.ttf");
const ITALIC: &[u8] = include_bytes!("/usr/share/fonts/noto/NotoSans-Italic.ttf");
const FONT_SYMBOLS: &[u8] = include_bytes!("/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf");
const FONT_SYMBOLS_2: &[u8] = include_bytes!("/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf");

const OUTPUT_STITCH_SIZE: UVec2 = UVec2 { x: 50, y: 70 };

const MMPI: f64 = 25.4;

const DPI: f64 = 300.0;

const DPMM: f64 = DPI / MMPI;

const PORTRAIT_SIZE: (Mm, Mm) = (Mm(210.0), Mm(297.0));

const IMAGE_PADDING: f64 = 5.0;

#[derive(Debug, Parser)]
pub struct Args {
    // Path to the input image
    #[arg(short)]
    input: PathBuf,
    // Path to the output image
    #[arg(short)]
    output: String,
    // Title of the document
    #[arg(short)]
    title: String,
    // The piece is by
    #[arg(short)]
    by: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input = {
        let bytes = fs::read(args.input)?;
        ::image::load_from_memory(&bytes)?
    };

    generate_pdf(&input, args.title, args.by)
        .save(&mut BufWriter::new(fs::File::create(args.output).unwrap()))?;

    Ok(())
}

fn generate_pdf(img: &DynamicImage, title: String, by: Option<String>) -> PdfDocumentReference {
    let (doc, curr_page, curr_layer) =
        PdfDocument::new(&title, PORTRAIT_SIZE.0, PORTRAIT_SIZE.1, "cover");
    let curr_layer = doc.get_page(curr_page).get_layer(curr_layer);

    let fonts = [
        (
            doc.add_external_font(std::io::Cursor::new(REGULAR))
                .unwrap(),
            REGULAR,
        ),
        (
            doc.add_external_font(std::io::Cursor::new(BOLD)).unwrap(),
            BOLD,
        ),
        (
            doc.add_external_font(std::io::Cursor::new(ITALIC)).unwrap(),
            ITALIC,
        ),
        (
            doc.add_external_font(std::io::Cursor::new(FONT_SYMBOLS))
                .unwrap(),
            FONT_SYMBOLS,
        ),
        (
            doc.add_external_font(std::io::Cursor::new(FONT_SYMBOLS_2))
                .unwrap(),
            FONT_SYMBOLS_2,
        ),
    ];

    let symbol_font_map = {
        let mut map: HashMap<_, _, RandomState> = HashMap::default();

        for c in SYMBOLS {
            for (font, font_bytes) in &fonts {
                if curr_layer.font_contains_char_glpyh(c, font) {
                    map.insert(c, (font.clone(), *font_bytes));
                }
            }
        }

        map
    };

    let floss_map = load_dmc_colors();

    // Set the pixels to the closest DMC colors
    let img = {
        let mut img = img.to_rgba8();
        for color in img.pixels_mut() {
            if color.0[3] == 0 {
                color.0[0] = 255;
                color.0[1] = 255;
                color.0[2] = 255;
                color.0[3] = 255;
                continue;
            }

            let lab_color = palette::Lab::<palette::white_point::D65, f64>::adapt_from(
                palette::rgb::Srgb::new(color.0[0], color.0[1], color.0[2]).into_format(),
            );

            let (_, selected_color) = floss_map
                .keys()
                .map(|color| {
                    (
                        palette::Lab::<palette::white_point::D65, f64>::adapt_from(
                            palette::rgb::Srgb::new(color.0[0], color.0[1], color.0[2])
                                .into_format(),
                        ),
                        color,
                    )
                })
                .min_by_key(|(lab, _)| float_ord::FloatOrd(lab.distance(lab_color)))
                .unwrap();

            *color = image::Rgba([
                selected_color.0[0],
                selected_color.0[1],
                selected_color.0[2],
                255,
            ]);
        }

        let img: DynamicImage = img.into();
        &img.to_rgb8().into()
    };

    let sub_images = sub_divide_images(img);
    let mut colors: HashMap<_, _, RandomState> = HashMap::default();

    for color in img.to_rgb8().pixels() {
        if color.0 == [255, 255, 255] {
            continue;
        }

        *colors.entry(*color).or_insert(0) += 1;
    }
    let total_pages =
        3 + if colors.len() <= 69 {
            1
        } else {
            ((colors.len() as f64 - 69.0) / 75.0).ceil() as usize + 1
        } + sub_images.len();

    let mut colors = colors
        .into_iter()
        .map(|(color, freq)| (color, freq, floss_map[&color]))
        .collect::<Vec<_>>();
    colors.sort_by_key(|(_, _, floss)| *floss);

    let color_symbol_map = colors
        .clone()
        .into_iter()
        .enumerate()
        .map(|(idx, (color, _, _))| (color, SYMBOLS[idx]))
        .collect::<HashMap<_, _>>();

    // Add border
    const BORDER_MARGIN: Mm = Mm(5.0);
    curr_layer.add_shape(Line {
        points: printpdf::calculate_points_for_rect(
            PORTRAIT_SIZE.0 - (BORDER_MARGIN * 2.0),
            PORTRAIT_SIZE.1 - (BORDER_MARGIN * 2.0),
            BORDER_MARGIN + ((PORTRAIT_SIZE.0 - (BORDER_MARGIN * 2.0)) / 2.0),
            BORDER_MARGIN + ((PORTRAIT_SIZE.1 - (BORDER_MARGIN * 2.0)) / 2.0),
        ),
        is_closed: true,
        has_fill: false,
        has_stroke: true,
        is_clipping_path: false,
    });

    // Add title text
    render_centered_text(
        &curr_layer,
        &title,
        30.0,
        (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(30.0)),
        &fonts[1],
    );

    // Add the by line
    let top_offset;
    if let Some(by) = &by {
        top_offset = 45.0;
        render_centered_text(
            &curr_layer,
            by,
            30.0,
            (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(45.0)),
            &fonts[2],
        );
    } else {
        top_offset = 42.0;
        render_centered_text(
            &curr_layer,
            "Original Pattern",
            24.0,
            (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(42.0)),
            &fonts[2],
        );
    }

    // Render Bottom Text
    let bottom_offset = 245.0;
    render_centered_text(
        &curr_layer,
        "Cross-Stitch Pattern",
        24.0,
        (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(250.0)),
        &fonts[0],
    );
    render_centered_text(
        &curr_layer,
        "BY",
        24.0,
        (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(260.0)),
        &fonts[0],
    );
    render_centered_text(
        &curr_layer,
        "needlethreading",
        24.0,
        (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(270.0)),
        &fonts[0],
    );

    // Render Page idx
    render_centered_text(
        &curr_layer,
        &format!("1 / {}", total_pages),
        18.0,
        (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(285.0)),
        &fonts[1],
    );

    // Adding the main image
    render_image_centered(
        curr_layer,
        img,
        BORDER_MARGIN.0,
        (PORTRAIT_SIZE.0 - BORDER_MARGIN).0,
        top_offset,
        bottom_offset,
        PORTRAIT_SIZE.1 .0,
    );

    if img.height() >= img.width() {
        let (curr_page, curr_layer) = doc.add_page(PORTRAIT_SIZE.0, PORTRAIT_SIZE.1, "preview");
        let layer = doc.get_page(curr_page).get_layer(curr_layer);

        // Render Page idx
        render_centered_text(
            &layer,
            &format!("2 / {}", total_pages),
            18.0,
            (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(290.0)),
            &fonts[1],
        );

        render_left_text(
            &layer,
            &title,
            16.0,
            (Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
            &fonts[0],
        );

        render_right_text(
            &layer,
            "needlethreading",
            16.0,
            (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
            &fonts[1],
        );

        render_image_centered(
            layer,
            img,
            0.0,
            PORTRAIT_SIZE.0 .0,
            10.0,
            PORTRAIT_SIZE.1 .0 - 10.0,
            PORTRAIT_SIZE.1 .0 - 5.0,
        );
    } else {
        let (curr_page, curr_layer) = doc.add_page(PORTRAIT_SIZE.1, PORTRAIT_SIZE.0, "preview");
        let layer = doc.get_page(curr_page).get_layer(curr_layer);

        // Render Page idx
        render_centered_text(
            &layer,
            &format!("2 / {}", total_pages),
            18.0,
            (PORTRAIT_SIZE.1 / 2.0, PORTRAIT_SIZE.0 - Mm(205.0)),
            &fonts[1],
        );

        render_ccw_rotated_start(&layer, &title, 24.0, (Mm(15.0), Mm(15.0)), &fonts[0]);

        render_ccw_rotated_end(
            &layer,
            "needlethreading",
            24.0,
            (Mm(15.0), PORTRAIT_SIZE.0 - Mm(15.0)),
            &fonts[1],
        );

        render_image_centered(
            layer,
            img,
            10.0,
            PORTRAIT_SIZE.1 .0,
            0.0,
            PORTRAIT_SIZE.0 .0 - 10.0,
            PORTRAIT_SIZE.0 .0 - 5.0,
        );
    }

    if img.height() >= img.width() {
        let (curr_page, curr_layer) = doc.add_page(PORTRAIT_SIZE.0, PORTRAIT_SIZE.1, "preview");
        let layer = doc.get_page(curr_page).get_layer(curr_layer);

        // Render Page idx
        render_centered_text(
            &layer,
            &format!("3 / {}", total_pages),
            18.0,
            (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(290.0)),
            &fonts[1],
        );

        render_left_text(
            &layer,
            &title,
            16.0,
            (Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
            &fonts[0],
        );

        render_right_text(
            &layer,
            "needlethreading",
            16.0,
            (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
            &fonts[1],
        );

        render_image_centered(
            layer.clone(),
            img,
            0.0,
            PORTRAIT_SIZE.0 .0,
            20.0,
            PORTRAIT_SIZE.1 .0,
            PORTRAIT_SIZE.1 .0,
        );

        draw_image_overlay(
            &layer,
            &img.to_rgb8(),
            UVec2::ZERO,
            0.0,
            PORTRAIT_SIZE.0 .0,
            20.0,
            PORTRAIT_SIZE.1 .0,
            PORTRAIT_SIZE.1 .0,
            &fonts,
            &color_symbol_map,
            &symbol_font_map,
        );
    } else {
        let (curr_page, curr_layer) = doc.add_page(PORTRAIT_SIZE.1, PORTRAIT_SIZE.0, "preview");
        let layer = doc.get_page(curr_page).get_layer(curr_layer);

        render_ccw_rotated_start(&layer, &title, 24.0, (Mm(15.0), Mm(15.0)), &fonts[0]);

        render_ccw_rotated_end(
            &layer,
            "needlethreading",
            24.0,
            (Mm(15.0), PORTRAIT_SIZE.0 - Mm(15.0)),
            &fonts[1],
        );

        render_image_centered(
            layer.clone(),
            img,
            10.0,
            PORTRAIT_SIZE.1 .0,
            0.0,
            PORTRAIT_SIZE.0 .0 - 10.0,
            PORTRAIT_SIZE.0 .0 - 5.0,
        );

        draw_image_overlay(
            &layer,
            &img.to_rgb8(),
            UVec2::ZERO,
            10.0,
            PORTRAIT_SIZE.1 .0,
            0.0,
            PORTRAIT_SIZE.0 .0 - 10.0,
            PORTRAIT_SIZE.0 .0 - 5.0,
            &fonts,
            &color_symbol_map,
            &symbol_font_map,
        );

        // Render Page idx
        render_centered_text(
            &layer,
            &format!("3 / {}", total_pages),
            18.0,
            (PORTRAIT_SIZE.1 / 2.0, PORTRAIT_SIZE.0 - Mm(205.0)),
            &fonts[1],
        );
    }

    // Generate the color count page
    let (curr_page, curr_layer) = doc.add_page(PORTRAIT_SIZE.0, PORTRAIT_SIZE.1, "threads");
    let layer = doc.get_page(curr_page).get_layer(curr_layer);

    render_left_text(
        &layer,
        &title,
        16.0,
        (Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
        &fonts[0],
    );

    render_right_text(
        &layer,
        "needlethreading",
        16.0,
        (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
        &fonts[1],
    );

    ruler(
        &layer,
        (Mm(10.0), PORTRAIT_SIZE.1 - Mm(18.0)),
        (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(18.0)),
    );

    semi_underlined_text(
        &layer,
        &format!("Dimension: {}w x {}h", img.width(), img.height()),
        0..9,
        (Mm(10.0), PORTRAIT_SIZE.1 - Mm(27.0)),
        18.0,
        &fonts[0],
    );

    semi_underlined_text(
        &layer,
        &format!(
            "Finished Size: {:.2} cm x {:.2} cm",
            (img.width() as f64 / 8.0) * 2.54,
            (img.height() as f64 / 8.0) * 2.54
        ),
        0..13,
        (Mm(10.0), PORTRAIT_SIZE.1 - Mm(37.0)),
        18.0,
        &fonts[0],
    );

    semi_underlined_text(
        &layer,
        "Cloth: Aida (16 t./inch)",
        0..5,
        (Mm(120.0), PORTRAIT_SIZE.1 - Mm(27.0)),
        18.0,
        &fonts[0],
    );

    semi_underlined_text(
        &layer,
        &format!("No. of colors: {} Colors", colors.len()),
        0..13,
        (Mm(120.0), PORTRAIT_SIZE.1 - Mm(37.0)),
        18.0,
        &fonts[0],
    );

    ruler(
        &layer,
        (Mm(10.0), PORTRAIT_SIZE.1 - Mm(43.0)),
        (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(43.0)),
    );

    // Render Page idx
    render_centered_text(
        &layer,
        &format!("4 / {}", total_pages),
        18.0,
        (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(285.0)),
        &fonts[1],
    );

    let mut top = Mm(50.0);
    let mut page_idx = 0;
    let mut row_idx = 0;
    let mut col_idx = 0;
    let mut layer = layer;

    let regular = doc
        .add_external_font(std::io::Cursor::new(REGULAR))
        .unwrap();

    for (idx, (color, freq, floss)) in colors.iter().enumerate() {
        layer.set_fill_color(printpdf::Color::Rgb(printpdf::Rgb {
            r: color.0[0] as f64 / 255.0,
            g: color.0[1] as f64 / 255.0,
            b: color.0[2] as f64 / 255.0,
            icc_profile: None,
        }));

        if ((PORTRAIT_SIZE.1 - top) - Mm(10.0 * row_idx as f64)).0 - 3.5 < 20.0 {
            row_idx = 0;
            col_idx += 1;
        }

        if col_idx > 2 {
            let (curr_page, curr_layer) =
                doc.add_page(PORTRAIT_SIZE.0, PORTRAIT_SIZE.1, "colors page");
            layer = doc.get_page(curr_page).get_layer(curr_layer);

            render_left_text(
                &layer,
                &title,
                16.0,
                (Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
                &fonts[0],
            );

            render_right_text(
                &layer,
                "needlethreading",
                16.0,
                (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
                &fonts[1],
            );

            ruler(
                &layer,
                (Mm(10.0), PORTRAIT_SIZE.1 - Mm(18.0)),
                (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(18.0)),
            );

            page_idx += 1;

            // Render Page idx
            render_centered_text(
                &layer,
                &format!("{} / {}", 4 + page_idx, total_pages),
                18.0,
                (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(285.0)),
                &fonts[1],
            );

            top = Mm(25.0);

            row_idx = 0;
            col_idx = 0;
        }

        layer.add_shape(Line {
            points: printpdf::calculate_points_for_rect(
                Mm(6.0),
                Mm(6.0),
                Mm(15.0) + Mm(65.0 * col_idx as f64),
                (PORTRAIT_SIZE.1 - top) - Mm(10.0 * row_idx as f64),
            ),
            is_closed: true,
            has_fill: true,
            has_stroke: true,
            is_clipping_path: false,
        });

        layer.add_shape(Line {
            points: printpdf::calculate_points_for_rect(
                Mm(10.0),
                Mm(6.0),
                Mm(25.0) + Mm(65.0 * col_idx as f64),
                (PORTRAIT_SIZE.1 - top) - Mm(10.0 * row_idx as f64),
            ),
            is_closed: true,
            has_fill: true,
            has_stroke: true,
            is_clipping_path: false,
        });

        let l = (0.2126 * (color.0[0] as f64 / 255.0).powf(2.2))
            + (0.7152 * (color.0[1] as f64 / 255.0).powf(2.2))
            + (0.0722 * (color.0[2] as f64 / 255.0).powf(2.2));

        if l > 0.5f64.powf(2.2) {
            layer.set_fill_color(printpdf::Color::Rgb(printpdf::Rgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                icc_profile: None,
            }));
        } else {
            layer.set_fill_color(printpdf::Color::Rgb(printpdf::Rgb {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                icc_profile: None,
            }));
        }

        render_centered_text(
            &layer,
            &format!("{}", SYMBOLS[idx]),
            12.0,
            (
                Mm(14.25) + Mm(65.0 * col_idx as f64),
                ((PORTRAIT_SIZE.1 - top) - Mm(1.5)) - Mm(10.0 * row_idx as f64),
            ),
            &symbol_font_map[&SYMBOLS[idx]],
        );

        layer.set_fill_color(printpdf::Color::Rgb(printpdf::Rgb {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            icc_profile: None,
        }));

        layer.use_text(
            format!("{} ({} ct)", floss, freq),
            16.0,
            Mm(32.0) + Mm(65.0 * col_idx as f64),
            ((PORTRAIT_SIZE.1 - top) - Mm(2.0)) - Mm(10.0 * row_idx as f64),
            &regular,
        );

        row_idx += 1;
    }

    // Generate pixel part pages
    for (idx, (sub_image, offset)) in sub_images.into_iter().enumerate() {
        let (curr_page, curr_layer) = doc.add_page(PORTRAIT_SIZE.0, PORTRAIT_SIZE.1, "threads");
        let layer = doc.get_page(curr_page).get_layer(curr_layer);

        render_left_text(
            &layer,
            &title,
            16.0,
            (Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
            &fonts[0],
        );

        render_right_text(
            &layer,
            "needlethreading",
            16.0,
            (PORTRAIT_SIZE.0 - Mm(10.0), PORTRAIT_SIZE.1 - Mm(15.0)),
            &fonts[1],
        );

        // Render Page idx
        render_centered_text(
            &layer,
            &format!("{} / {}", (4 + page_idx) + idx + 1, total_pages),
            18.0,
            (PORTRAIT_SIZE.0 / 2.0, PORTRAIT_SIZE.1 - Mm(285.0)),
            &fonts[1],
        );

        render_image_centered(
            layer.clone(),
            &sub_image.clone().into(),
            0.0,
            PORTRAIT_SIZE.0 .0,
            0.0,
            PORTRAIT_SIZE.1 .0 - 40.0,
            PORTRAIT_SIZE.1 .0 - 20.0,
        );

        draw_image_overlay(
            &layer,
            &sub_image,
            offset,
            0.0,
            PORTRAIT_SIZE.0 .0,
            0.0,
            PORTRAIT_SIZE.1 .0 - 40.0,
            PORTRAIT_SIZE.1 .0 - 20.0,
            &fonts,
            &color_symbol_map,
            &symbol_font_map,
        );
    }

    doc
}

fn render_centered_text(
    layer: &PdfLayerReference,
    text: &str,
    size: f64,
    center_position: (Mm, Mm),
    font: &(IndirectFontRef, &[u8]),
) {
    let width = {
        let font = rusttype::Font::try_from_bytes(font.1).unwrap();

        font.layout(
            text,
            rusttype::Scale {
                x: size as f32,
                y: size as f32,
            },
            rusttype::Point { x: 0.0, y: 0.0 },
        )
        .last()
        .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
        .unwrap_or(0.0)
    } / 2.1;

    layer.begin_text_section();
    layer.use_text(
        text,
        size,
        center_position.0 - (Mm(width as f64) / 2.0),
        center_position.1,
        &font.0,
    );
    layer.end_text_section();
}

fn render_left_text(
    layer: &PdfLayerReference,
    text: &str,
    size: f64,
    start_position: (Mm, Mm),
    font: &(IndirectFontRef, &[u8]),
) {
    layer.begin_text_section();
    layer.use_text(text, size, start_position.0, start_position.1, &font.0);
    layer.end_text_section();
}

fn render_right_text(
    layer: &PdfLayerReference,
    text: &str,
    size: f64,
    start_position: (Mm, Mm),
    font: &(IndirectFontRef, &[u8]),
) {
    let width = {
        let font = rusttype::Font::try_from_bytes(font.1).unwrap();

        font.layout(
            text,
            rusttype::Scale {
                x: size as f32,
                y: size as f32,
            },
            rusttype::Point { x: 0.0, y: 0.0 },
        )
        .last()
        .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
        .unwrap_or(0.0)
    } / 2.1;

    layer.begin_text_section();
    layer.use_text(
        text,
        size,
        start_position.0 - Mm(width as f64),
        start_position.1,
        &font.0,
    );
    layer.end_text_section();
}

fn render_ccw_rotated_start(
    layer: &PdfLayerReference,
    text: &str,
    size: f64,
    start_position: (Mm, Mm),
    font: &(IndirectFontRef, &[u8]),
) {
    layer.begin_text_section();
    layer.set_font(&font.0, size);
    layer.set_text_cursor(Mm(0.0), Mm(0.0));
    layer.set_text_matrix(printpdf::TextMatrix::TranslateRotate(
        start_position.0.into_pt(),
        start_position.1.into_pt(),
        90.0,
    ));
    layer.write_text(text, &font.0);
    layer.end_text_section();
}

fn render_ccw_rotated_end(
    layer: &PdfLayerReference,
    text: &str,
    size: f64,
    start_position: (Mm, Mm),
    font: &(IndirectFontRef, &[u8]),
) {
    let width = {
        let font = rusttype::Font::try_from_bytes(font.1).unwrap();

        font.layout(
            text,
            rusttype::Scale {
                x: size as f32,
                y: size as f32,
            },
            rusttype::Point { x: 0.0, y: 0.0 },
        )
        .last()
        .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
        .unwrap_or(0.0)
    } / 2.1;

    layer.begin_text_section();
    layer.set_font(&font.0, size);
    layer.set_text_cursor(Mm(0.0), Mm(0.0));
    layer.set_text_matrix(printpdf::TextMatrix::TranslateRotate(
        start_position.0.into_pt(),
        (start_position.1 - Mm(width as f64)).into_pt(),
        90.0,
    ));
    layer.write_text(text, &font.0);
    layer.end_text_section();
}

fn render_ccw_rotated_centered(
    layer: &PdfLayerReference,
    text: &str,
    size: f64,
    start_position: (Mm, Mm),
    font: &(IndirectFontRef, &[u8]),
) {
    let width = {
        let font = rusttype::Font::try_from_bytes(font.1).unwrap();

        font.layout(
            text,
            rusttype::Scale {
                x: size as f32,
                y: size as f32,
            },
            rusttype::Point { x: 0.0, y: 0.0 },
        )
        .last()
        .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
        .unwrap_or(0.0)
    } / 2.1;

    layer.begin_text_section();
    layer.set_font(&font.0, size);
    layer.set_text_cursor(Mm(0.0), Mm(0.0));
    layer.set_text_matrix(printpdf::TextMatrix::TranslateRotate(
        start_position.0.into_pt(),
        (start_position.1 - Mm(width as f64 / 2.0)).into_pt(),
        90.0,
    ));
    layer.write_text(text, &font.0);
    layer.end_text_section();
}

fn render_image_centered(
    layer: PdfLayerReference,
    img: &DynamicImage,
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
    height: f64,
) {
    let (img, translate) = {
        let size = DVec2 {
            x: img.width() as f64,
            y: img.height() as f64,
        };
        let screen_size = DVec2 {
            x: right - (left + IMAGE_PADDING * 2.0),
            y: bottom - (top + IMAGE_PADDING * 2.0),
        } * DPMM;
        let mut scale = (screen_size / size).min_element() as u32;

        if scale > 58 {
            scale = 58;
        }

        let img = img.resize(
            img.width() * scale,
            img.height() * scale,
            image::imageops::FilterType::Nearest,
        );

        let translate = (screen_size - (size * scale as f64)) / 2.0;

        (
            img,
            (
                (translate.x / DPMM) + left + IMAGE_PADDING,
                (translate.y / DPMM) + (height - bottom) + IMAGE_PADDING,
            ),
        )
    };
    printpdf::Image::from_dynamic_image(&img).add_to_layer(
        layer,
        ImageTransform {
            translate_x: Some(Mm(translate.0)),
            translate_y: Some(Mm(translate.1)),
            dpi: Some(DPI),
            ..Default::default()
        },
    );
}

fn draw_image_overlay(
    layer: &PdfLayerReference,
    img: &RgbImage,
    offset: UVec2,
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
    height: f64,
    fonts: &[(IndirectFontRef, &[u8])],
    color_symbol_map: &HashMap<Rgb<u8>, char>,
    symbol_font_map: &HashMap<char, (IndirectFontRef, &[u8])>,
) {
    const GRID: UVec2 = UVec2 { x: 10, y: 10 };
    let image_size = UVec2 {
        x: img.width(),
        y: img.height(),
    };

    let (scaled_image_size, step_size, translate, x_extra, y_extra) = {
        let size = image_size.as_dvec2();
        let screen_size = DVec2 {
            x: right - (left + IMAGE_PADDING * 2.0),
            y: bottom - (top + IMAGE_PADDING * 2.0),
        } * DPMM;
        let mut scale = (screen_size / size).min_element() as u32;

        if scale > 58 {
            scale = 58;
        }

        let translate = (screen_size - (size * scale as f64)) / 2.0;

        (
            (size * scale as f64) / DPMM,
            (GRID * scale).as_dvec2() / DPMM,
            (
                (translate.x / DPMM) + left + IMAGE_PADDING,
                (translate.y / DPMM) + (height - bottom) + IMAGE_PADDING,
            ),
            ((image_size.x % GRID.x) * scale) as f64 / DPMM,
            ((image_size.y % GRID.y) * scale) as f64 / DPMM,
        )
    };

    layer.set_outline_thickness(0.1);
    layer.set_outline_color(printpdf::Color::Rgb(printpdf::Rgb {
        r: 0.388,
        g: 0.388,
        b: 0.388,
        icc_profile: None,
    }));

    let inner_step_size = step_size / GRID.as_dvec2();
    for i in 0..image_size.x {
        layer.add_shape(Line {
            points: vec![
                (
                    Point::new(
                        Mm(translate.0 + inner_step_size.x * i as f64),
                        Mm(translate.1),
                    ),
                    true,
                ),
                (
                    Point::new(
                        Mm(translate.0 + inner_step_size.x * i as f64),
                        Mm(translate.1 + scaled_image_size.y),
                    ),
                    true,
                ),
            ],
            is_closed: false,
            has_fill: false,
            has_stroke: true,
            is_clipping_path: false,
        });
    }

    for i in 0..image_size.y {
        layer.add_shape(Line {
            points: vec![
                (
                    Point::new(
                        Mm(translate.0),
                        Mm(translate.1 + inner_step_size.y * i as f64),
                    ),
                    true,
                ),
                (
                    Point::new(
                        Mm(translate.0 + scaled_image_size.x),
                        Mm(translate.1 + inner_step_size.y * i as f64),
                    ),
                    true,
                ),
            ],
            is_closed: false,
            has_fill: false,
            has_stroke: true,
            is_clipping_path: false,
        });
    }

    let sections = image_size / GRID;

    layer.set_outline_thickness(1.0);
    layer.set_outline_color(printpdf::Color::Rgb(printpdf::Rgb {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        icc_profile: None,
    }));

    for i in 1..=sections.x {
        layer.add_shape(Line {
            points: vec![
                (
                    Point::new(Mm(translate.0 + step_size.x * i as f64), Mm(translate.1)),
                    true,
                ),
                (
                    Point::new(
                        Mm(translate.0 + step_size.x * i as f64),
                        Mm(translate.1 + scaled_image_size.y),
                    ),
                    true,
                ),
            ],
            is_closed: false,
            has_fill: false,
            has_stroke: true,
            is_clipping_path: false,
        });

        render_centered_text(
            &layer,
            &format!("{}", 10 * i + offset.x * OUTPUT_STITCH_SIZE.x),
            8.0,
            (
                Mm(translate.0 + step_size.x * i as f64),
                Mm(translate.1 + scaled_image_size.y) + Mm(1.0),
            ),
            &fonts[1],
        );
    }

    let rem = image_size % GRID;
    if rem.x != 0 {
        let extra = if offset.x * OUTPUT_STITCH_SIZE.x > 99 {
            4.0
        } else {
            2.0
        };
        render_centered_text(
            &layer,
            &format!("{}", offset.x * OUTPUT_STITCH_SIZE.x + image_size.x),
            8.0,
            (
                Mm((translate.0 + step_size.x * (sections.x as f64 + 1.0)).min(
                    translate.0 + scaled_image_size.x + if x_extra < extra { extra } else { 0.0 },
                )),
                Mm(translate.1 + scaled_image_size.y) + Mm(1.0),
            ),
            &fonts[1],
        );
    }

    for i in 0..sections.y {
        layer.add_shape(Line {
            points: vec![
                (
                    Point::new(
                        Mm(translate.0),
                        Mm(translate.1 + step_size.y * i as f64 + y_extra),
                    ),
                    true,
                ),
                (
                    Point::new(
                        Mm(translate.0 + scaled_image_size.x),
                        Mm(translate.1 + step_size.y * i as f64 + y_extra),
                    ),
                    true,
                ),
            ],
            is_closed: false,
            has_fill: false,
            has_stroke: true,
            is_clipping_path: false,
        });

        render_ccw_rotated_centered(
            layer,
            &format!(
                "{}",
                10 * (sections.y - i) + offset.y * OUTPUT_STITCH_SIZE.y
            ),
            8.0,
            (
                Mm(translate.0 - 1.0),
                Mm(translate.1 + step_size.y * i as f64 + y_extra),
            ),
            &fonts[1],
        );
    }

    let rem = image_size % GRID;
    if rem.y != 0 {
        let extra = if image_size.y > 99 { 4.0 } else { 2.0 };
        render_ccw_rotated_centered(
            &layer,
            &format!("{}", offset.y * OUTPUT_STITCH_SIZE.y + image_size.y),
            8.0,
            (
                Mm(translate.0 - 1.0),
                Mm((translate.1 - (step_size.y - y_extra))
                    .max(translate.1 - if y_extra < extra { extra } else { 0.0 })),
            ),
            &fonts[1],
        );
    }

    // Add thick lines around the border
    layer.add_shape(Line {
        points: vec![
            (Point::new(Mm(translate.0), Mm(translate.1)), true),
            (
                Point::new(Mm(translate.0), Mm(translate.1 + scaled_image_size.y)),
                true,
            ),
        ],
        is_closed: false,
        has_fill: false,
        has_stroke: true,
        is_clipping_path: false,
    });
    layer.add_shape(Line {
        points: vec![
            (
                Point::new(Mm(translate.0 + scaled_image_size.x), Mm(translate.1)),
                true,
            ),
            (
                Point::new(
                    Mm(translate.0 + scaled_image_size.x),
                    Mm(translate.1 + scaled_image_size.y),
                ),
                true,
            ),
        ],
        is_closed: false,
        has_fill: false,
        has_stroke: true,
        is_clipping_path: false,
    });
    layer.add_shape(Line {
        points: vec![
            (Point::new(Mm(translate.0), Mm(translate.1)), true),
            (
                Point::new(Mm(translate.0 + scaled_image_size.x), Mm(translate.1)),
                true,
            ),
        ],
        is_closed: false,
        has_fill: false,
        has_stroke: true,
        is_clipping_path: false,
    });
    layer.add_shape(Line {
        points: vec![
            (
                Point::new(Mm(translate.0), Mm(translate.1 + scaled_image_size.y)),
                true,
            ),
            (
                Point::new(
                    Mm(translate.0 + scaled_image_size.x),
                    Mm(translate.1 + scaled_image_size.y),
                ),
                true,
            ),
        ],
        is_closed: false,
        has_fill: false,
        has_stroke: true,
        is_clipping_path: false,
    });

    // Generate color markers
    for y in 0..image_size.y {
        for x in 0..image_size.x {
            let color = img.get_pixel(x, y);

            if color.0 == [255, 255, 255] {
                continue;
            }

            let l = (0.2126 * (color.0[0] as f64 / 255.0).powf(2.2))
                + (0.7152 * (color.0[1] as f64 / 255.0).powf(2.2))
                + (0.0722 * (color.0[2] as f64 / 255.0).powf(2.2));

            if l > 0.5f64.powf(2.2) {
                layer.set_fill_color(printpdf::Color::Rgb(printpdf::Rgb {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    icc_profile: None,
                }));
            } else {
                layer.set_fill_color(printpdf::Color::Rgb(printpdf::Rgb {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    icc_profile: None,
                }));
            }

            render_centered_text(
                &layer,
                &format!("{}", color_symbol_map[color]),
                inner_step_size.y * 2.0,
                (
                    Mm(translate.0
                        + inner_step_size.x * x as f64
                        + (inner_step_size.x * 0.43211062)),
                    PORTRAIT_SIZE.1
                        - Mm(top
                            + translate.1
                            + inner_step_size.y * y as f64
                            + (inner_step_size.y * 0.720184367)),
                ),
                &symbol_font_map[&color_symbol_map[color]],
            );
        }
    }
}

fn ruler(layer: &PdfLayerReference, start: (Mm, Mm), end: (Mm, Mm)) {
    layer.add_shape(Line {
        points: vec![
            (
                Point {
                    x: start.0.into_pt(),
                    y: start.1.into_pt(),
                },
                true,
            ),
            (
                Point {
                    x: end.0.into_pt(),
                    y: end.1.into_pt(),
                },
                true,
            ),
        ],
        is_closed: false,
        has_fill: false,
        has_stroke: true,
        is_clipping_path: false,
    });
}

fn semi_underlined_text(
    layer: &PdfLayerReference,
    text: &str,
    underline_chars: Range<usize>,
    start_position: (Mm, Mm),
    size: f64,
    font: &(IndirectFontRef, &[u8]),
) {
    let (start, end) = {
        let font = rusttype::Font::try_from_bytes(font.1).unwrap();

        let mut layout = font.layout(
            text,
            rusttype::Scale {
                x: size as f32,
                y: size as f32,
            },
            rusttype::Point { x: 0.0, y: 0.0 },
        );

        let start = if underline_chars.start == 0 {
            0.0
        } else {
            layout
                .clone()
                .nth(underline_chars.start - 1)
                .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
                .unwrap_or(0.0)
        };
        let end = layout
            .nth(underline_chars.end - 1)
            .map(|g| g.position().x + g.unpositioned().h_metrics().advance_width)
            .unwrap_or(0.0);

        (start / 2.1, end / 2.1)
    };

    layer.begin_text_section();
    layer.use_text(text, size, start_position.0, start_position.1, &font.0);
    ruler(
        layer,
        (
            start_position.0 + Mm(start as f64),
            start_position.1 - Mm(1.0),
        ),
        (
            start_position.0 + Mm(end as f64),
            start_position.1 - Mm(1.0),
        ),
    );
    layer.end_text_section();
}

fn load_dmc_colors() -> HashMap<Rgb<u8>, usize> {
    #[derive(serde::Deserialize)]
    struct DmcColor {
        floss: Option<usize>,
        red: u8,
        green: u8,
        blue: u8,
    }

    let colors: Vec<DmcColor> =
        serde_json::from_str(include_str!("../../dmc_colors.json")).unwrap();

    colors
        .into_iter()
        .filter_map(
            |DmcColor {
                 floss,
                 red,
                 green,
                 blue,
             }| floss.map(|floss| (Rgb::from([red, green, blue]), floss)),
        )
        .collect()
}

fn sub_divide_images(img: &DynamicImage) -> Vec<(RgbImage, UVec2)> {
    let img = img.to_rgb8();
    let mut images = Vec::default();

    for j in 0..((img.height() / OUTPUT_STITCH_SIZE.y)
        + if img.height() % OUTPUT_STITCH_SIZE.y != 0 {
            1
        } else {
            0
        })
    {
        for i in 0..((img.width() / OUTPUT_STITCH_SIZE.x)
            + if img.width() % OUTPUT_STITCH_SIZE.x != 0 {
                1
            } else {
                0
            })
        {
            images.push((
                img.view(
                    i * OUTPUT_STITCH_SIZE.x,
                    j * OUTPUT_STITCH_SIZE.y,
                    if (i * OUTPUT_STITCH_SIZE.x + OUTPUT_STITCH_SIZE.x) > img.width() {
                        img.width() % OUTPUT_STITCH_SIZE.x
                    } else {
                        OUTPUT_STITCH_SIZE.x
                    },
                    if (j * OUTPUT_STITCH_SIZE.y + OUTPUT_STITCH_SIZE.y) > img.height() {
                        img.height() % OUTPUT_STITCH_SIZE.y
                    } else {
                        OUTPUT_STITCH_SIZE.y
                    },
                )
                .to_image(),
                UVec2 { x: i, y: j },
            ));
        }
    }

    images
}
