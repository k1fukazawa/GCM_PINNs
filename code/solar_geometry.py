"""Solar input features from the time and position only (NOAA approximation).

geometry() returns cos SZA and the sine and cosine of the local mean solar time
(LMT) on a latitude-longitude grid, as an array of shape (3, nlat, nlon).
"""
import calendar
from datetime import datetime, timedelta, timezone
import numpy as np
import torch

FEATURE_NAMES = ("cos_sza", "sin_local_mean_time", "cos_local_mean_time")


def grid_centers(lat_idx, lon_idx, nlat, nlon):
    return (90.0 - 180.0 * (np.asarray(lat_idx) + 0.5) / nlat,
            360.0 * (np.asarray(lon_idx) + 0.5) / nlon)


def geometry(hour, lat_deg, lon_deg, time_origin="2020-01-01T00:00:00+00:00"):
    origin = datetime.fromisoformat(time_origin.replace("Z", "+00:00"))
    if origin.tzinfo is None or origin.utcoffset() != timedelta(0):
        raise ValueError("time_origin needs an explicit UTC offset of +00:00")
    if not np.isfinite(hour):
        raise ValueError("hour must be finite")
    date = origin.astimezone(timezone.utc) + timedelta(hours=float(hour))
    utc = date.hour + date.minute / 60 + (date.second + date.microsecond / 1e6) / 3600
    gamma = 2 * np.pi / (366 if calendar.isleap(date.year) else 365) * (
        date.timetuple().tm_yday - 1 + (utc - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                   - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    dec = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
           - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
           - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))[:, None]
    lon = np.asarray(lon_deg, dtype=np.float64)[None, :]
    lmst = (utc + lon / 15) % 24
    ha = np.deg2rad((utc*60 + 4*lon + eq) / 4 - 180)
    cosz = np.clip(np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(ha), -1, 1)
    phase = 2*np.pi*lmst/24
    shape = cosz.shape
    features = np.stack((cosz, np.broadcast_to(np.sin(phase), shape),
                         np.broadcast_to(np.cos(phase), shape))).astype(np.float32)
    return features


def append_solar(x, hour, lat, lon, origin, data_start_hour=0):
    f = torch.as_tensor(geometry(data_start_hour + hour, lat, lon, origin),
                        dtype=x.dtype, device=x.device)
    f = f[None, :, None].expand(x.shape[0], 3, x.shape[2], x.shape[3], x.shape[4])
    return torch.cat((x, f), dim=1)
