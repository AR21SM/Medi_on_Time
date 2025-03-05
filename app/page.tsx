import Image from "next/image";
import axios from "axios";

async function getuserdata(){
  // await new Promise((r) => setTimeout(r, 5000)); 
  const response=await axios.get("http://localhost:3000/api/user")

  return response.data;
}
// since 
export default async function Home() {
  // const userdetails=await getuserdata();
  return (
    <div>
    hello bhai
    </div>
    // <div className="flex flex-col justify-center h-screen">
    // <div className="flex justify-center">
    //     <div className="border p-8 rounded">
    //         <div>
    //             Name: {userdetails.name}
    //         </div>
    //         {userdetails.email}
    //     </div>
    // </div>
// </div>
  );
} 
