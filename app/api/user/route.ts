
import { PrismaClient } from '@prisma/client';
import { NextRequest, NextResponse } from 'next/server';

const client= new PrismaClient();

export async function POST(req:NextRequest){
    const body=await req.json();
    const user= await client.user.create({
        data:{
            username:body.username,
            password:body.password,
        }
    })
    console.log("heloohelloooooo")
    console.log(user.id);
    return Response.json({
        message:"you are logged in"
    })
}