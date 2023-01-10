mod contract_tests;
mod error;
mod execute;
pub mod helpers;
pub mod msg;
mod query;
pub mod state;

use serde::{Deserialize, Serialize};
pub use crate::error::ContractError;
pub use crate::msg::{ExecuteMsg, InstantiateMsg, MintMsg, MinterResponse, QueryMsg};
pub use crate::state::Cw721Contract;
use cosmwasm_std::Empty;
use opencv::core::{ add_weighted, bitwise_and, Mat, no_array, Vector};
use opencv::imgcodecs::{imencode, imread, IMREAD_UNCHANGED, IMWRITE_PNG_COMPRESSION};
use opencv::imgproc::{COLOR_BGRA2GRAY, COLOR_GRAY2BGRA, cvt_color, THRESH_BINARY_INV, threshold};

use rand::distributions::WeightedIndex;
use rand::{thread_rng};
use rand::prelude::*;
use schemars::JsonSchema;
use crate::ImageType::{ACC, BEARD, EARS, EYES, FACE, HAIR, MOUTH, NOSE};

// This is a simple type to let us handle empty extensions
pub type Extension = Option<Empty>;

#[derive(Serialize, Deserialize, Clone, Debug, Copy, JsonSchema)]
pub struct NftImage {
    acc: u8,
    beard: u8,
    ears: u8,
    eyes: u8,
    face: u8,
    hair: u8,
    mouth: u8,
    nose: u8,
}

enum ImageType {
    ACC,
    BEARD,
    EARS,
    EYES,
    FACE,
    HAIR,
    MOUTH,
    NOSE,
}
impl NftImage {
    /// acc (2)
    /// beard (4)
    /// ears (4)
    /// eyes (5)
    /// face (2)
    /// hair (12)
    /// mouth (6)
    /// nose (2)
    fn new() -> Self {
        NftImage {
            acc: random_num(ImageType::ACC),
            beard: random_num(ImageType::BEARD),
            ears: random_num(ImageType::EARS),
            eyes: random_num(ImageType::EYES),
            face: random_num(ImageType::FACE),
            hair: random_num(ImageType::HAIR),
            mouth: random_num(ImageType::MOUTH),
            nose: random_num(ImageType::NOSE),
        }
    }
}

impl<'a, 'b> PartialEq for NftImage {
    fn eq(&self, other: &Self) -> bool {
        if
        self.eyes == other.eyes
            && self.acc == other.acc
            && self.face == other.face
            && self.beard == other.beard
            && self.hair == other.hair
            && self.ears == other.ears
            && self.mouth == other.mouth
            && self.nose == other.nose {
            return true;
        }
        false
    }
}


// 이미지 합성
fn alpha_composite(background_image: &Mat, overlay_image: Mat) -> Mat {

    // GRAYSCALE 타입으로 변경
    let mut overlay_image_gray = Mat::default();
    cvt_color(&overlay_image, &mut overlay_image_gray, COLOR_BGRA2GRAY, 0).unwrap();
    // println!("{:?}",mat_to_base64(&overlay_image_gray));

    // 이미지 흑백처리
    let mut overlay_image_mask_gray = Mat::default();
    threshold(&overlay_image_gray, &mut overlay_image_mask_gray, 1.0, 255.0, THRESH_BINARY_INV).unwrap();
    // println!("{:?}",mat_to_base64(&overlay_image_mask_gray));

    // BGRA 타입으로 다시 변경
    let mut overlay_image_mask = Mat::default();
    cvt_color(&overlay_image_mask_gray, &mut overlay_image_mask, COLOR_GRAY2BGRA, 0).unwrap();

    // 배경과 흑백처리된 오버레이 이미지 합성
    let mut background_image_bg = Mat::default();
    bitwise_and(background_image, &overlay_image_mask, &mut background_image_bg, &no_array()).unwrap();
    // println!("{:?}",mat_to_base64(&background_image_bg));

    // 최종본
    let mut result = Mat::default();
    add_weighted(&background_image_bg, 0.95, &overlay_image, 1.0, -10.0, &mut result, 0).unwrap();
    // if weighted_check {
    // } else {
    //     add(&background_image_bg, overlay_image, &mut result, &no_array(), 0).unwrap();
    // };
    result
}

// 랜덤 이미지 리턴
#[warn(unused_mut)]
fn random_num(nft_image_type: ImageType) -> u8 {
    let acc_arr: [(u8, u8); 2] = [
        (1, 10),
        (2, 20)
    ];
    let beard_arr: [(u8, u8); 4] = [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 1)
    ];
    let ears_arr: [(u8, u8); 4] = [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 1)
    ];
    let eyes_arr: [(u8, u8); 5] = [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 1),
        (5, 80)
    ];
    let face_arr: [(u8, u8); 2] = [
        (1, 10),
        (2, 20),
    ];
    let hair_arr: [(u8, u8); 12] = [
        (1, 10),
        (2, 10),
        (3, 10),
        (4, 1),
        (5, 10),
        (6, 10),
        (7, 11),
        (8, 24),
        (9, 4),
        (10, 19),
        (11, 10),
        (12, 22)
    ];
    let mouse_arr: [(u8, u8); 6] = [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 1),
        (5, 80),
        (6, 4),
    ];
    let nose_arr: [(u8, u8); 2] = [
        (1, 10),
        (2, 20),
    ];
    let type_arr: [Vec<(u8, u8)>; 8] = [Vec::from(acc_arr), Vec::from(beard_arr), Vec::from(ears_arr), Vec::from(eyes_arr), Vec::from(face_arr)
        , Vec::from(hair_arr), Vec::from(mouse_arr), Vec::from(nose_arr)];

    match nft_image_type {
        ImageType::ACC => {
            let type_code = 0;
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
        ImageType::BEARD => {
            let type_code = 1;
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
        ImageType::EARS => {
            let type_code = 2;
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
        ImageType::EYES => {
            let type_code = 3;
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
        ImageType::FACE => {
            let type_code = 4;
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
        ImageType::HAIR => {
            let type_code = 5;
            // println!("{:?}",type_arr[type_code].iter().map(|item| item.1));
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
        ImageType::MOUTH => {
            let  type_code = 6;
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
        ImageType::NOSE => {
            let type_code = 7;
            let dist = WeightedIndex::new(type_arr[type_code].iter().map(|item| item.1)).unwrap();
            type_arr[type_code][dist.sample(&mut thread_rng())].0
        }
    }
}

fn get_image(image_type: ImageType, num: u8) -> Mat {
    let mut path = "./images/face_parts".to_string();

    match image_type {
        ImageType::ACC => {
            path = path.to_owned() + "/access/acc";
        }
        ImageType::BEARD => {
            path = path.to_owned() + "/beard/beard";
        }
        ImageType::EARS => {
            path = path.to_owned() + "/ears/ears";
        }
        ImageType::EYES => {
            path = path.to_owned() + "/eyes/eyes";
        }
        ImageType::FACE => {
            path = path.to_owned() + "/face/face";
        }
        ImageType::HAIR => {
            path = path.to_owned() + "/hair/hair";
        }
        ImageType::MOUTH => {
            path = path.to_owned() + "/mouth/m";
        }
        ImageType::NOSE => {
            path = path.to_owned() + "/nose/n";
        }
    }
    path = path.to_owned() + &num.to_string() + ".png";
    // println!("{}",path);
    imread(&path, IMREAD_UNCHANGED).unwrap()
}

fn get_nft_image(nft_image: &NftImage) -> Mat {
    let face = get_image(FACE, nft_image.face);
    let eyes = get_image(EYES, nft_image.eyes);
    let nose = get_image(NOSE, nft_image.nose);
    let mouth = get_image(MOUTH, nft_image.mouth);
    let ears = get_image(EARS, nft_image.ears);
    let hair = get_image(HAIR, nft_image.hair);
    let beard = get_image(BEARD, nft_image.beard);
    let acc = get_image(ACC, nft_image.acc);

    let image = alpha_composite(&face, eyes);
    let image = alpha_composite(&image, nose);
    let image = alpha_composite(&image, mouth);
    let image = alpha_composite(&image, ears);
    let image = alpha_composite(&image, hair);
    let image = alpha_composite(&image, beard);
    alpha_composite(&image, acc)
}


#[warn(dead_code)]
fn mat_to_base64(img: &Mat) -> String {
    let mut image_vector: Vector<u8> = Vector::new();
    let param: Vector<i32> = Vector::from(vec![IMWRITE_PNG_COMPRESSION, 3]);
    // println!("{:?}",img);
    imencode(".png", img, &mut image_vector, &param).unwrap();
    let image_vec = image_vector.to_vec();
    // img.write_to(&mut Cursor::new(&mut image_data), ImageOutputFormat::Png)
    //     .unwrap();


    let res_base64: String = base64::encode(image_vec);
    format!("data:image/png;base64,{}", res_base64)
}

#[cfg(not(feature = "library"))]
pub mod entry {
    use super::*;

    use cosmwasm_std::entry_point;
    use cosmwasm_std::{Binary, Deps, DepsMut, Env, MessageInfo, Response, StdResult};

    // This makes a conscious choice on the various generics used by the contract
    #[entry_point]
    pub fn instantiate(
        deps: DepsMut,
        env: Env,
        info: MessageInfo,
        msg: InstantiateMsg,
    ) -> StdResult<Response> {
        let tract = Cw721Contract::<Extension, Empty>::default();
        tract.instantiate(deps, env, info, msg)
    }

    #[entry_point]
    pub fn execute(
        deps: DepsMut,
        env: Env,
        info: MessageInfo,
        msg: ExecuteMsg<Extension>,
    ) -> Result<Response, ContractError> {
        let tract = Cw721Contract::<Extension, Empty>::default();
        tract.execute(deps, env, info, msg)
    }

    #[entry_point]
    pub fn query(deps: Deps, env: Env, msg: QueryMsg) -> StdResult<Binary> {
        let tract = Cw721Contract::<Extension, Empty>::default();
        tract.query(deps, env, msg)
    }
}
